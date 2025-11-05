"""
Rig Preparation Detection Launcher
Choose your performance level for optimal FPS
"""

import sys
import subprocess

def show_menu():
    print("\n" + "="*50)
    print("RIG PREPARATION DETECTION SYSTEM")
    print("="*50)
    print("Choose performance level:")
    print()
    print("1. ULTRA FAST  - Maximum FPS (~15-30 FPS)")
    print("   • Motion-based detection")
    print("   • Minimal processing")
    print("   • Best for real-time monitoring")
    print()
    print("2. OPTIMIZED   - Balanced (~10-20 FPS)")
    print("   • HOG person detection")
    print("   • Frame skipping + scaling")
    print("   • Good accuracy + speed")
    print()
    print("3. STANDARD    - Full accuracy (~5-10 FPS)")
    print("   • Full HOG detection")
    print("   • Maximum accuracy")
    print("   • Slower but more reliable")
    print()
    print("4. Exit")
    print("="*50)

def main():
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                print("\nStarting ULTRA FAST mode...")
                print("Expected FPS: 15-30")
                subprocess.run([sys.executable, "rig_prep_detection_ultra_fast.py"])
                
            elif choice == "2":
                print("\nStarting OPTIMIZED mode...")
                print("Expected FPS: 10-20")
                subprocess.run([sys.executable, "rig_prep_detection_optimized.py"])
                
            elif choice == "3":
                print("\nStarting STANDARD mode...")
                print("Expected FPS: 5-10")
                subprocess.run([sys.executable, "rig_prep_detection_opencv.py"])
                
            elif choice == "4":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice! Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
