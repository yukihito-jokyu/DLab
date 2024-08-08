from firebase_config import get_firestore_db
from firebase_admin import storage

def update_status(model_id, status):
    db = get_firestore_db()
    
    # コレクションから該当するドキュメントを取得
    doc_ref = db.collection('model_management').document(model_id)
    
    # ドキュメントの存在を確認してから更新
    doc = doc_ref.get()
    if doc.exists:
        doc_ref.update({'status': status})
        print(f'\nDocument with model_id {model_id} status updated to {status}.\n')
    else:
        print(f'\nDocument with model_id {model_id} does not exist.\n')

def save_result_manegement(model_id, accuracy, loss):
    db = get_firestore_db()
    
    # コレクションから該当するドキュメントを取得
    doc_ref = db.collection('model_management').document(model_id)
    
    # ドキュメントの存在を確認してから更新
    doc = doc_ref.get()
    if doc.exists:
        doc_ref.update({
                'accuracy': accuracy,
                'loss': loss
            })
        print(f'\nDocument with model_id {model_id} result updated to Acc:{accuracy} Loss:{loss}.\n')
    else:
        print(f'\nDocument with model_id {model_id} does not exist.\n')

def save_result_readarboard(project_name, user_id, user_name, accuracy):
    db = get_firestore_db()
    
    # コレクションから該当するドキュメントを取得
    doc_ref = db.collection(f'{project_name}_reader_board').document(user_id)
    
    # ドキュメントの存在を確認
    doc = doc_ref.get()

    if doc.exists:
        existing_data = doc.to_dict()
        existing_accuracy = existing_data.get('accuracy', 0)
        
        # 新しいaccuracyが高い場合にのみ上書き
        if accuracy > existing_accuracy:
            doc_ref.update({
                'user_name': user_name,
                'accuracy': accuracy
            })
            print(f"\nUpdated {user_id} with new accuracy {accuracy}\n")
        else:
            print(f"\nNo update performed for {user_id} because existing accuracy {existing_accuracy} is higher or equal.\n")
    else:
        # ドキュメントが存在しない場合は新規作成
        doc_ref.set({
            'user_id': user_id,
            'user_name': user_name,
            'accuracy': accuracy
        })
        print(f"\nCreated new document for {user_id} with accuracy {accuracy}\n")

# ファイルをダウンロードする関数
def download_file(source_blob_name, destination_file_name):
    try:
        bucket = storage.bucket()
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"\nDownloaded {source_blob_name} to {destination_file_name}.\n")
        return destination_file_name
    except Exception as e:
        print(f"\nFailed to download {source_blob_name}: {e}\n")
        return None

# ファイルをアップロードする関数
def upload_file(local_file_path, storage_blob_path):
    try:
        bucket = storage.bucket()
        blob = bucket.blob(storage_blob_path)
        blob.upload_from_filename(local_file_path)
        print(f"\nUploaded {local_file_path} to {storage_blob_path}.\n")
        return {"message": "successfully"}
    except Exception as e:
        return {"message": str(e)}