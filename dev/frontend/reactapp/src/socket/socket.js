import { io } from "socket.io-client";

let socket = io('ws://127.0.0.1:5000', {
  // reconnectionAttempts: 1,  // 再接続の試行回数
  // reconnectionDelay: 5000   // 再接続の試行間隔（ミリ秒）
  ransports: ['websocket']
});

// let socket = null

// ソケット接続のエラーハンドリング
// socket.on('connect_error', (err) => {
//   // エラーをコンソールに表示しないようにする
//   console.error = () => {};
//   // socketをnullに置き換える
//   socket = null;
//   // ユーザーにエラー通知を表示する
//   // alert('サーバーとの接続に問題があります。しばらくしてから再試行してください。');
// });


export { socket };