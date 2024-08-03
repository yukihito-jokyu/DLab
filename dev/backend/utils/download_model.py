import os
import shutil
import zipfile
from datetime import datetime

# 指定したモデルディレクトリをダウンロードする関数
def download_model_directories(data):
    user_id = data["user_id"]
    project_name = data["Project_name"]
    model_id_list = data["model_id_list"]

    base_path = "./user"
    user_path = os.path.join(base_path, user_id)
    project_path = os.path.join(user_path, project_name)
    
    zip_file_paths = []
    
    try:
        # 各モデルのディレクトリを処理
        for model_id in model_id_list:
            model_path = os.path.join(project_path, model_id)
            
            # __pycache__ を削除
            pycache_path = os.path.join(model_path, '__pycache__')
            if os.path.exists(pycache_path):
                shutil.rmtree(pycache_path)
            
            # モデルのディレクトリをZIPファイルに圧縮
            zip_file_path = os.path.join(project_path, f"{model_id}.zip")
            shutil.make_archive(zip_file_path.replace('.zip', ''), 'zip', model_path)
            zip_file_paths.append(zip_file_path)
        
        # ユーザーのダウンロードフォルダのパスを取得
        downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
        
        # 日付と時間を取得し＆ファイル名を生成
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_zip_file_name = f"model_{current_time}.zip"
        final_zip_file = os.path.join(downloads_folder, final_zip_file_name)
        
        # すべてのZIPファイルを1つのZIPファイルに集約
        with zipfile.ZipFile(final_zip_file, 'w') as zipf:
            for file in zip_file_paths:
                zipf.write(file, os.path.basename(file))
                os.remove(file)

        return {"message": "successfully"}
    except Exception as e:
        return {"message": str(e)}
    