import os
import shutil

# 指定されたユーザーのmodelディレクトリを削除する関数
def delete_model_directories(data):
    user_id = data["user_id"]
    project_name = data["Project_name"]
    model_id_list = data["model_id_list"]

    base_path = "./user"
    project_path = os.path.join(base_path, user_id, project_name)
    
    all_deleted = True
    
    try:
        for model_id in model_id_list:
            model_path = os.path.join(project_path, model_id)
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            else:
                all_deleted = False
        if all_deleted:
            return {"message": "successfully"}
        else:
            return {"message": "Failed"}
    except Exception as e:
        return {"message": str(e)}
