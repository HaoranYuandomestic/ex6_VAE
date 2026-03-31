"""
选择保留指定数量的文件脚本
保留 val2014 文件夹中的 10000 个文件
保留 wikiart_subset 文件夹中的 8000 个文件
"""

import os
import random
import shutil
from pathlib import Path

def select_and_keep_files(folder_path, num_files_to_keep):
    """
    从指定文件夹中随机选择指定数量的文件，删除其他文件
    
    Args:
        folder_path: 文件夹路径
        num_files_to_keep: 要保留的文件数量
    """
    folder_path = Path(folder_path)
    
    # 获取文件夹中的所有文件
    all_files = [f for f in folder_path.iterdir() if f.is_file()]
    
    print(f"\n处理文件夹: {folder_path}")
    print(f"当前文件总数: {len(all_files)}")
    print(f"要保留的文件数: {num_files_to_keep}")
    
    # 如果文件数小于等于要保留的数量，则不需要删除
    if len(all_files) <= num_files_to_keep:
        print(f"文件数量已经不超过 {num_files_to_keep}，不需要处理")
        return
    
    # 随机选择要保留的文件
    files_to_keep = set(random.sample(all_files, num_files_to_keep))
    
    # 删除不在保留列表中的文件
    files_to_delete = [f for f in all_files if f not in files_to_keep]
    
    print(f"将删除 {len(files_to_delete)} 个文件")
    
    for file_path in files_to_delete:
        try:
            file_path.unlink()  # 删除文件
            if len(files_to_delete) % 1000 == 0:
                print(f"已删除 {len(files_to_delete)} 个文件...")
        except Exception as e:
            print(f"删除文件失败 {file_path}: {e}")
    
    print(f"完成！保留了 {num_files_to_keep} 个文件")

def main():
    """主函数"""
    # 设置文件夹路径
    script_dir = Path(__file__).parent
    val2014_folder = script_dir / "val2014"
    wikiart_folder = script_dir / "wikiart_subset"
    
    # 配置参数
    val2014_keep = 10000
    wikiart_keep = 8000
    
    print("=" * 60)
    print("文件选择脚本")
    print("=" * 60)
    
    # 处理 val2014 文件夹
    if val2014_folder.exists():
        select_and_keep_files(val2014_folder, val2014_keep)
    else:
        print(f"错误: {val2014_folder} 文件夹不存在")
    
    # 处理 wikiart_subset 文件夹
    if wikiart_folder.exists():
        select_and_keep_files(wikiart_folder, wikiart_keep)
    else:
        print(f"错误: {wikiart_folder} 文件夹不存在")
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)

if __name__ == "__main__":
    # 设置随机种子以保证可重现性（可选）
    random.seed(42)
    main()
