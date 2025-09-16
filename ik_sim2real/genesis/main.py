#!/usr/bin/env python3
"""
Main program for Piper IK Sim2Real control
Simple demonstration of end-effector control using Genesis IK solver
"""

import argparse
import numpy as np
import time
from rich.console import Console
from piper_ik_controller import PiperIKSim2Real

console = Console()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Piper IK Sim2Real Control")
    parser.add_argument("--real", action="store_true", default=False,
                       help="Use real robot hardware (default: simulation only)")
    parser.add_argument("--no-viewer", action="store_true", default=False,
                       help="Disable Genesis viewer")
    parser.add_argument("--interface", type=str, default="can0",
                       help="CAN interface name for real robot (default: can0)")
    args = parser.parse_args()

    console.print("[bold blue]Starting Piper IK Sim2Real Control[/bold blue]")
    
    # Initialize controller with user parameters
    try:
        controller = PiperIKSim2Real(
            use_real_robot=args.real,
            interface_name=args.interface,
            show_viewer=not args.no_viewer
        )
    except Exception as e:
        console.print(f"[red]Failed to initialize controller: {e}[/red]")
        return

    # Start control system
    controller.start_control()
    
    try:
        # Demo sequence: move to different target positions
        demo_sequence = [
            {"pos": [0.06, 0.002, 0.25], "quat": [0.707, 0, 0.707, 0], "grasp": False, "duration": 6},
            # {"pos": [0.2, 0.002, 0.20], "quat": [0.707, 0, 0.707, 0], "grasp": False, "duration": 6},
            # {"pos": [0.2, 0.1, 0.20], "quat": [0.707, 0, 0.707, 0], "grasp": False, "duration": 6},
            {"pos": [0.32304699588668623, -0.05111836931472475, 0.19899168059757344], "quat": [0.707, 0, 0.707, 0], "grasp": False, "duration": 30},
            # {"pos": [0.2, -0.1, 0.20], "quat": [0.707, 0, 0.707, 0], "grasp": False, "duration": 30},
        ]
        
        console.print("[green]Starting demo sequence...[/green]")
        
        for i, target in enumerate(demo_sequence, 1):
            console.print(f"[blue]Step {i}: Moving to position {target['pos']}[/blue]")
            
            controller.set_target_pose(
                position=target["pos"],
                quaternion=target["quat"],
                grasp=target["grasp"]
            )
            
            time.sleep(target["duration"])
            
        console.print("[green]Demo sequence completed[/green]")
        
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Demo error: {e}[/red]")
    finally:
        # Clean shutdown
        console.print("[blue]Stopping control system...[/blue]")
        controller.stop_control()
        console.print("[green]Program finished[/green]")

if __name__ == "__main__":
    main() 