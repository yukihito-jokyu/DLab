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