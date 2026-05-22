import ijson
import time
import os

def count_json_list_items(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}")
        return

    file_size_gb = os.path.getsize(file_path) / (1024 ** 3)
    print(f"📂 开始处理文件: {file_path} (约 {file_size_gb:.2f} GB)")
    print("⏳ 正在流式读取，请稍候...")
    
    count = 0
    start_time = time.time()

    try:
        # 必须以二进制模式 ('rb') 打开文件，ijson 需要字节流
        with open(file_path, 'rb') as f:
            # ijson.items 中的 'item' 专门用于指代根级别列表下的子元素
            for item in ijson.items(f, 'item'):
                count += 1
                
                # 每扫描 10 万个 item 打印一次进度，避免控制台输出过多影响性能
                if count % 100000 == 0:
                    elapsed = time.time() - start_time
                    print(f"   -> 已扫描 {count} 个 item... (耗时 {elapsed:.2f} 秒)")
                    
    except Exception as e:
        print(f"\n❌ 解析过程中发生错误: {e}")
        return

    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n✅ 解析完成！")
    print("-" * 30)
    print(f"📊 总 Item 数量: {count}")
    print(f"⏱️  总耗时: {total_time:.2f} 秒")
    print("-" * 30)

if __name__ == "__main__":
    # 目标文件路径
    TARGET_FILE = "/gemini/space/zyf/FG-CLIP/data/FineHARD/output.json"
    
    count_json_list_items(TARGET_FILE)