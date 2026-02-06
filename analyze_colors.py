import numpy as np

# ==========================================
# 🚑 紧急修复: 给 colormath 库打补丁
# ==========================================
def patch_asscalar(a):
    return a.item()
setattr(np, "asscalar", patch_asscalar)

# 补丁打完后再引入 colormath
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import itertools
import os

# ================= 配置区域 =================

# 打印参数
LAYER_HEIGHT = 0.08  # 层高
LAYERS = 5           # 混色层数
BACKING_COLOR = np.array([255, 255, 255]) # 底板颜色 (白色)

# 耗材定义
FILAMENTS = {
    0: {"name": "White (Jade)", "rgb": [255, 255, 255], "td": 5.0},   # 白色
    1: {"name": "Cyan",         "rgb": [0, 134, 214],   "td": 3.5},   # 青色
    2: {"name": "Magenta",      "rgb": [236, 0, 140],   "td": 3.0},   # 品红
    3: {"name": "Green",        "rgb": [0, 174, 66],    "td": 2.0},   # 拓竹绿
    4: {"name": "Yellow",       "rgb": [244, 238, 42],  "td": 6.0},   # 黄色
    
    # === 关键数据 ===
    5: {"name": "Black",        "rgb": [0, 0, 0],       "td": 0.2},   # 黑色 (遮光剂)
    6: {"name": "Red",          "rgb": [193, 46, 31],   "td": 4.0},   # 红色
    7: {"name": "Deep Blue",    "rgb": [10, 41, 137],   "td": 2.3},   # 深蓝
}

# 色差阈值
THRESHOLD_DELTA_E = 2.5 

# ===========================================

def calculate_alpha(td_value, layer_height):
    blending_distance = td_value / 10.0
    if blending_distance <= 0: return 1.0
    alpha = layer_height / blending_distance
    return min(max(alpha, 0.0), 1.0)

def mix_colors(stack):
    """
    stack: [底层 ... 顶层]
    stack[0] = Layer 5 (背面/遮光层)
    stack[4] = Layer 1 (正面/观赏层)
    """
    current_rgb = BACKING_COLOR.astype(float)
    for fid in stack:
        fil = FILAMENTS[fid]
        f_rgb = np.array(fil["rgb"])
        f_alpha = calculate_alpha(fil["td"], LAYER_HEIGHT)
        current_rgb = f_rgb * f_alpha + current_rgb * (1.0 - f_alpha)
    return current_rgb.astype(np.uint8)

def rgb_to_lab(rgb):
    rgb_obj = sRGBColor(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
    return convert_color(rgb_obj, LabColor)

def main():
    COLOR_COUNT = 8 
    
    print(f"🔄 开始模拟 {COLOR_COUNT}色 {LAYERS}层 全排列 ({COLOR_COUNT**LAYERS} 种组合)...")
    print(f"📏 色差阈值 (Delta E): {THRESHOLD_DELTA_E}")
    print(f"🧱 物理约束: Black TD = {FILAMENTS[5]['td']} (强遮盖)")
    
    all_combinations = []
    permutations = itertools.product(range(COLOR_COUNT), repeat=LAYERS)
    
    dropped_by_rule = 0

    for stack in permutations:
        
        # ==================== 🛡️ V9 最终法则：把黑色赶到后面去！ ====================
        
        # 1. 检查正面 (Layer 1 / Stack[4])
        # 这里的 5 是黑色。如果 Stack[4] 是黑，说明第一层是黑。
        if stack[4] == 5:
            # 【豁免条款】只有全黑 [5,5,5,5,5] 这个基准点可以保留
            if set(stack) == {5}:
                pass 
            # 其他所有“表面黑”的组合，统统删掉！
            else:
                dropped_by_rule += 1
                continue 
        
        # 2. 背面 (Layer 5 / Stack[0])
        # 这里完全不管！允许它是 5 (黑色)。
        # 这样算法就会自动把黑色安排在背面，用来做阴影。
        
        # ===============================================================

        final_rgb = mix_colors(stack)
        all_combinations.append({
            "stack": stack,
            "rgb": final_rgb,
            "lab": rgb_to_lab(final_rgb)
        })
        
    print(f"🚫 已剔除 {dropped_by_rule} 个“正面含脏黑”组合。")
    print(f"✅ 计算完成，剩余 {len(all_combinations)} 个有效组合。")
    print("🧹 开始执行视觉去重筛选...")
    
    unique_colors = []
    total = len(all_combinations)
    
    for i, candidate in enumerate(all_combinations):
        is_distinct = True
        for existing in unique_colors:
            delta_e = delta_e_cie2000(candidate["lab"], existing["lab"])
            if delta_e < THRESHOLD_DELTA_E:
                is_distinct = False
                break
        if is_distinct:
            unique_colors.append(candidate)
            
        if i % 5000 == 0:
            print(f"   处理进度: {i}/{total} | 当前保留: {len(unique_colors)}")

    total_combinations = COLOR_COUNT ** LAYERS

    print("-" * 30)
    print(f"🎉 最终结果: 在 {total_combinations} 种组合中")
    print(f"💎 肉眼可见的独立颜色数量: {len(unique_colors)}")
    print(f"📉 冗余率: {(1 - len(unique_colors)/total_combinations)*100:.1f}%")

    # ================= 37x37 布局 (容量 2738) =================
    target_count = 2738  # 37 * 37 * 2
    output_dir = "assets"
    
    print("-" * 30)
    print(f"💾 Saving top {target_count} colors to '{output_dir}/'...")
    
    final_selection = unique_colors
    
    # 填充不足的部分
    if len(final_selection) < target_count:
        # 用黄色(4)填充空位
        dummy_stack = [4] * LAYERS 
        while len(final_selection) < target_count:
            final_selection.append({"stack": dummy_stack})
    else:
        final_selection = final_selection[:target_count]
    
    stacks_data = [item["stack"] for item in final_selection]
    stacks_array = np.array(stacks_data, dtype=np.uint8)
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    save_path = os.path.join(output_dir, "smart_8color_stacks.npy")
    np.save(save_path, stacks_array)
    print(f"✅ Saved to '{save_path}' (Capacity: {len(stacks_array)})")
    
    if len(unique_colors) <= 1024:
        print("💡 结论: 1024 个色块完全足够覆盖所有颜色变化！")
    else:
        print(f"💡 结论: 颜色变化丰富，建议使用双板打印。")

if __name__ == "__main__":
    main()