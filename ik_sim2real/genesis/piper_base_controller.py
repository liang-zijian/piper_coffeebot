import numpy as np
import time
from rich.console import Console
from rich.table import Table
from rich.text import Text

try:
    from piper_sdk import C_PiperInterface_V2
    PIPER_SDK_AVAILABLE = True
except ImportError:
    PIPER_SDK_AVAILABLE = False
    C_PiperInterface_V2 = None

console = Console()

def enable_fun(piper):
    """Enable the robotic arm and check enable status"""
    enable_flag = False
    timeout = 5
    start_time = time.time()
    
    while not enable_flag:
        elapsed_time = time.time() - start_time
        
        # Get enable status
        arm_info = piper.GetArmLowSpdInfoMsgs()
        motor_status = [
            arm_info.motor_1.foc_status.driver_enable_status,
            arm_info.motor_2.foc_status.driver_enable_status,
            arm_info.motor_3.foc_status.driver_enable_status,
            arm_info.motor_4.foc_status.driver_enable_status,
            arm_info.motor_5.foc_status.driver_enable_status,
            arm_info.motor_6.foc_status.driver_enable_status
        ]
        
        enable_flag = all(motor_status)
        
        # Clear screen and display status
        console.clear()
        table = Table(title=f"Enable Status ({elapsed_time:.1f}s)", show_header=True)
        table.add_column("Motor", style="cyan")
        table.add_column("Status")
        
        for i, status in enumerate(motor_status):
            status_text = Text("✓", style="green") if status else Text("✗", style="red")
            table.add_row(f"Motor {i+1}", status_text)
        
        console.print(table)
        
        piper.EnableArm(7)
        piper.GripperCtrl(0, 1000, 0x01, 0)
        
        if elapsed_time > timeout:
            console.print("[red]Enable timeout, program exit[/red]")
            time.sleep(2)
            exit(0)
        
        time.sleep(0.5)


class PiperBaseController:
    """Base controller for Piper robotic arm with real robot control isolation"""
    
    def __init__(self, use_real_robot=False, interface_name="can0"):
        """
        Initialize base controller
        
        Args:
            use_real_robot (bool): Whether to use real robot hardware (default: False)
            interface_name (str): CAN interface name for real robot
        """
        self.use_real_robot = use_real_robot
        self.interface_name = interface_name
        self.piper_real = None
        self.is_enabled = False
        
        # Initialize real robot interface only when needed
        if self.use_real_robot:
            if not PIPER_SDK_AVAILABLE:
                raise ImportError("piper_sdk is not available. Cannot use real robot control.")
            self.piper_real = C_PiperInterface_V2(interface_name)
            console.print(f"[green]Real robot interface initialized with {interface_name}[/green]")
        else:
            console.print("[yellow]Running in simulation mode only[/yellow]")

    def enable_control(self):
        """Enable real-time control"""
        if not self.use_real_robot:
            console.print("[yellow]Simulation mode: skipping real robot enable[/yellow]")
            self.is_enabled = True
            return
            
        self.piper_real.ConnectPort()
        self.piper_real.GripperCtrl(0, 1000, 0x00, 0)
        time.sleep(1.5)
        enable_fun(self.piper_real)
        self.is_enabled = True
        console.print("[green]Real robot control enabled[/green]")

    def reset_to_initial_positions(self):
        """Reset the robot to its initial positions"""
        if not self.use_real_robot:
            console.print("[yellow]Simulation mode: skipping real robot reset[/yellow]")
            return
            
        self.move_joint_real([0] * 6)
        self.piper_real.GripperCtrl(0, 1000, 0x01, 0)
        time.sleep(1)
        console.print("[green]Reset to initial positions complete[/green]")

    def move_joint_real(self, target_joints, grasp=False):
        """
        Move joints to target positions in real-time
        
        Args:
            target_joints (list): Target joint angles in radians
            grasp (bool): Whether to close gripper
        """
        if not self.use_real_robot:
            return
            
        if not self.is_enabled:
            console.print("[red]Error: Robot not enabled[/red]")
            return
            
        # Control gripper
        if grasp:
            self.piper_real.GripperCtrl(0, 1000, 0x01, 0)
        else:
            self.piper_real.GripperCtrl(5000 * 1000, 1000, 0x01, 0)

        # Convert radians to motor units
        factor = 57295.7795
        joints = [0] * 6
        for i in range(6):
            joints[i] = round(target_joints[i] * factor)
            
        self.piper_real.MotionCtrl_2(0x01, 0x01, 60, 0x00)
        
        try:
            self.piper_real.JointCtrl(joints[0], joints[1], joints[2], joints[3], joints[4], joints[5])
        except Exception as e:
            console.print(f"[red]Error moving joints: {e}[/red]")
            return

    def control_gripper(self, grasp=False):
        """
        Control gripper state
        
        Args:
            grasp (bool): True to close gripper, False to open
        """
        if not self.use_real_robot:
            return
            
        if grasp:
            self.piper_real.GripperCtrl(0, 1000, 0x01, 0)
        else:
            self.piper_real.GripperCtrl(5000 * 1000, 1000, 0x01, 0)

    def disconnect(self):
        """Disconnect from real robot"""
        if self.use_real_robot and self.piper_real:
            self.reset_to_initial_positions()
            console.print("[green]Real robot disconnected[/green]")