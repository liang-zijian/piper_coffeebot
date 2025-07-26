import numpy as np
import genesis as gs
import threading
import time
from scipy.spatial.transform import Rotation as R
from rich.console import Console
from piper_base_controller import PiperBaseController

console = Console()

class PiperIKSim2Real(PiperBaseController):
    """
    Piper IK controller that inherits from base controller and adds 
    end-effector control using Genesis IK solver
    """
    
    def __init__(self, use_real_robot=False, interface_name="can0", show_viewer=True):
        """
        Initialize IK controller with simulation
        
        Args:
            use_real_robot (bool): Whether to use real robot hardware
            interface_name (str): CAN interface name for real robot
            show_viewer (bool): Whether to show Genesis viewer
        """
        super().__init__(use_real_robot, interface_name)
        
        self.show_viewer = show_viewer
        self.scene = None
        self.piper_robot = None
        self.cams = []
        self.lock = threading.Lock()
        self.running = True
        
        # Control parameters
        self.target_pos = None
        self.target_quat = None
        self.is_grasp = False
        
        # Joint configuration
        self.JOINT_NAMES = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7', 'joint8'
        ]
        self.dofs_idx = None
        
        # Initialize simulation
        self._init_simulation()

    def _init_simulation(self):
        """Initialize Genesis simulation environment"""
        console.print("[blue]Initializing Genesis simulation...[/blue]")
        
        gs.init(backend=gs.gpu)

        viewer_options = gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
            max_FPS=60,
        )

        self.scene = gs.Scene(
            viewer_options=viewer_options,
            rigid_options=gs.options.RigidOptions(dt=0.01),
            show_viewer=self.show_viewer,
            show_FPS=False
        )

        # Add cameras
        cam_0 = self.scene.add_camera(res=(1280, 960), pos=(0, 0.5, 1.5), lookat=(0.5, 0.5, 0.0), fov=55, GUI=False)
        cam_1 = self.scene.add_camera(res=(1280, 960), pos=(1.5, -0.5, 1.5), lookat=(0.5, 0.5, 0), fov=55, GUI=False)
        cam_2 = self.scene.add_camera(res=(1280, 960), pos=(0.5, 1.0, 1.5), lookat=(0.5, 0.5, 0.0), fov=55, GUI=False)
        self.cams = [cam_0, cam_1, cam_2]

        # Load Piper robot (simulation)
        self.piper_robot = self.scene.add_entity(
            gs.morphs.MJCF(file="/home/ubuntu/workspace/piper_ws/ik_sim2real/genesis/agilex_piper/piper.xml")
        )

        # Build scene
        self.scene.build()

        # Set initial pose
        self.dofs_idx = [self.piper_robot.get_joint(name).dof_idx_local for name in self.JOINT_NAMES]
        
        initial_joints = np.array([0, 0, 0, 0, 0, 0, 0.04, -0.04])
        self.piper_robot.control_dofs_position(initial_joints, self.dofs_idx)
        
        # Set control parameters
        self.piper_robot.set_dofs_kv(np.array([750, 750, 750, 750, 700, 700, 10, 10]))
        self.piper_robot.set_dofs_kp(np.array([6000, 6000, 6000, 6000, 6000, 6000, 200, 200]))
        
        console.print("[green]Genesis simulation initialized successfully[/green]")

    def start_control(self):
        """Start the control system"""
        # Enable real robot if needed
        if self.use_real_robot:
            self.enable_control()
        
        # Start control loop
        threading.Thread(target=self._control_loop, daemon=True).start()
        console.print("[green]Control system started[/green]")

    def _control_loop(self):
        """Main control loop running in separate thread"""
        while self.running:
            try:
                with self.lock:
                    if self.target_pos is not None:
                        self._update_end_effector_control()
                    self._run_simulation_step()
                time.sleep(0.05)  # 20Hz control loop
            except Exception as e:
                console.print(f"[red]Control loop error: {e}[/red]")
                break

    def _update_end_effector_control(self):
        """Update end-effector control using IK"""
        try:
            # Get end-effector link
            end_effector = self.piper_robot.get_link("link6")
            
            # Visualize current pose
            self._visualize_end_effector_pose(end_effector)
            
            # Get current position for fallback
            current_pos = end_effector.get_pos().cpu().numpy().copy()
            
            # Set default orientation if not specified
            target_quat = self.target_quat if self.target_quat is not None else np.array([0.707, 0, 0.707, 0])
            
            # Solve inverse kinematics
            new_qpos = self.piper_robot.inverse_kinematics(
                link=end_effector,
                pos=self.target_pos,
                quat=target_quat,
                max_solver_iters=60
            )
            
            if new_qpos is not None:
                # Update simulation joints
                self._update_joint_positions(new_qpos)
                
                # Control real robot if enabled
                if self.use_real_robot:
                    real_joints = new_qpos[:6].cpu().numpy()
                    self.move_joint_real(real_joints, self.is_grasp)
                    
                    
        except Exception as e:
            console.print(f"[red]IK solver failed: {e}[/red]")

    def _visualize_end_effector_pose(self, end_effector):
        """Visualize end-effector coordinate frame"""
        try:
            self.scene.clear_debug_objects()
            
            link_pos_tensor = end_effector.get_pos()
            link_quat_tensor = end_effector.get_quat()  # w, x, y, z
            
            if link_pos_tensor is not None and link_quat_tensor is not None:
                link_pos = link_pos_tensor.cpu().numpy()
                link_quat = link_quat_tensor.cpu().numpy()

                # Convert Genesis quaternion (w,x,y,z) to scipy format (x,y,z,w)
                rot = R.from_quat([link_quat[1], link_quat[2], link_quat[3], link_quat[0]])
                
                axis_length = 0.1
                x_axis = rot.apply([1, 0, 0])
                y_axis = rot.apply([0, 1, 0])
                z_axis = rot.apply([0, 0, 1])
                
                # Draw coordinate frame
                self.scene.draw_debug_arrow(pos=link_pos, vec=x_axis * axis_length, color=(1, 0, 0, 1))  # Red X
                self.scene.draw_debug_arrow(pos=link_pos, vec=y_axis * axis_length, color=(0, 1, 0, 1))  # Green Y
                self.scene.draw_debug_arrow(pos=link_pos, vec=z_axis * axis_length, color=(0, 0, 1, 1))  # Blue Z
                
        except Exception as e:
            console.print(f"[yellow]Cannot visualize end-effector pose: {e}[/yellow]")

    def _update_joint_positions(self, target_qpos):
        """Update simulation joint positions"""
        target_qpos_copy = target_qpos.clone()
        
        # Control gripper based on grasp state
        if self.is_grasp:
            target_qpos_copy[-2:] = 0.0  # Close gripper
        else:
            target_qpos_copy[-2] = 0.04   # Open gripper
            target_qpos_copy[-1] = -0.04
        
        # Update arm joints (first 6 joints)
        self.piper_robot.control_dofs_position(target_qpos_copy[:-2], self.dofs_idx[:-2])
        
        # Update gripper joints
        self.piper_robot.control_dofs_position(target_qpos_copy[-2:], self.dofs_idx[-2:])

    def _run_simulation_step(self):
        """Run one simulation step"""
        self.scene.step()

    def set_target_pose(self, position, quaternion=None, grasp=False):
        """
        Set target end-effector pose
        
        Args:
            position (np.array): Target position [x, y, z]
            quaternion (np.array): Target quaternion [w, x, y, z] (optional)
            grasp (bool): Whether to close gripper
        """
        with self.lock:
            self.target_pos = np.array(position, dtype=np.float32)
            if quaternion is not None:
                self.target_quat = np.array(quaternion, dtype=np.float32)
            self.is_grasp = grasp
            
        console.print(f"[blue]Target pose set: pos={position}, grasp={grasp}[/blue]")

    def get_current_pose(self):
        """
        Get current end-effector pose
        
        Returns:
            tuple: (position, quaternion) of end-effector
        """
        try:
            end_effector = self.piper_robot.get_link("link6")
            pos = end_effector.get_pos().cpu().numpy()
            quat = end_effector.get_quat().cpu().numpy()  # w, x, y, z format
            return pos, quat
        except Exception as e:
            console.print(f"[red]Error getting current pose: {e}[/red]")
            return None, None

    def stop_control(self):
        """Stop the control system"""
        self.running = False
        
        # Reset to initial positions
        if self.use_real_robot:
            self.reset_to_initial_positions()
        
        console.print("[yellow]Control system stopped[/yellow]")

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_control()
        if self.use_real_robot:
            self.disconnect() 