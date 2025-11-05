"""
Performance Comparison and Recommendations
for Rig Preparation Detection System
"""

def show_performance_guide():
    print("\n" + "="*60)
    print("RIG PREPARATION DETECTION - PERFORMANCE GUIDE")
    print("="*60)
    
    print("\n📊 PERFORMANCE COMPARISON:")
    print("-" * 40)
    print("Mode         | FPS Range | Accuracy | Use Case")
    print("-" * 40)
    print("Ultra Fast   | 15-30     | Medium   | Real-time monitoring")
    print("Optimized    | 10-20     | Good     | Balanced performance")
    print("Standard     | 5-10      | High     | Maximum accuracy")
    print("-" * 40)
    
    print("\n🚀 OPTIMIZATION TECHNIQUES USED:")
    print("• Frame skipping (process every N frames)")
    print("• Image scaling (smaller detection resolution)")
    print("• Optimized HOG parameters")
    print("• Background subtraction (ultra-fast mode)")
    print("• Reduced camera resolution")
    print("• Minimal buffer size")
    
    print("\n⚙️ RECOMMENDATIONS:")
    print("• For live monitoring: Use ULTRA FAST mode")
    print("• For good balance: Use OPTIMIZED mode")
    print("• For best accuracy: Use STANDARD mode")
    print("• If still slow: Check camera drivers/hardware")
    
    print("\n🔧 ADDITIONAL OPTIMIZATIONS:")
    print("• Close other applications")
    print("• Use dedicated camera (not webcam)")
    print("• Ensure good lighting")
    print("• Position camera to minimize false positives")
    
    print("\n💡 HOTKEYS AVAILABLE:")
    print("• 'q' - Quit application")
    print("• 'r' - Reset ROI selection")
    print("• 's' - Toggle performance settings (optimized mode)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    show_performance_guide()
    
    input("\nPress Enter to continue...")
