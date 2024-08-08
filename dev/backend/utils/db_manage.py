from firebase_config import get_firestore_db

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
            print(f"Updated {user_id} with new accuracy {accuracy}")
        else:
            print(f"No update performed for {user_id} because existing accuracy {existing_accuracy} is higher or equal.")
    else:
        # ドキュメントが存在しない場合は新規作成
        doc_ref.set({
            'user_id': user_id,
            'user_name': user_name,
            'accuracy': accuracy
        })
        print(f"Created new document for {user_id} with accuracy {accuracy}")
