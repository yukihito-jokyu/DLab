import os

# ユーザーのディレクトリを作成する関数
def create_user_directory(data):
    
    user_id = data["user_id"]

    base_path = "./user"
    user_path = os.path.join(base_path, user_id)
    
    try:
        print('yes')
        os.makedirs(user_path, exist_ok=True)
        os.makedirs(os.path.join(user_path, "CartPole"), exist_ok=True)
        os.makedirs(os.path.join(user_path, "FlappyBird"), exist_ok=True)
        return {"message": "successfully"}
    except Exception as e:
        print('no')
        print(e)
        return {"message": str(e)}



# プロジェクトディレクトリを作成する関数
def create_project_directory(data):

    user_id = data["user_id"]
    project_name = data["project_name"]

    base_path = "./user"
    user_path = os.path.join(base_path, user_id)
    project_path = os.path.join(user_path, project_name)
    
    try:
        os.makedirs(project_path, exist_ok=True)
        return {"message": "successfully"}
    except Exception as e:
        return {"message": str(e)}



# モデルディレクトリを作成する関数
def create_model_directory_from_dict(data):

    user_id = data["user_id"]
    project_name = data["project_name"]
    model_id = data["model_id"]

    base_path = "./user"
    project_path = os.path.join(base_path, user_id, project_name)
    model_path = os.path.join(project_path, model_id)
    
    try:
        os.makedirs(model_path, exist_ok=True)
        return {"message": "successfully"}
    except Exception as e:
        return {"message": str(e)}
