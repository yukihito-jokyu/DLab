const ip = require('ip');
const express = require("express");
const cors = require('cors');
const bodyParser = require('body-parser');
const app = express();
const mysql = require('mysql');

app.use(cors());
app.use(bodyParser.json());

const PORT = 4000;
// const blockedIP = "::ffff:192.168.0.12"; // ブロックするIPアドレス

// // IPブロックのミドルウェア
// const blockIPMiddleware = (req, res, next) => {
//     const clientIP = req.ip;
//     console.log(clientIP);
//     if (clientIP === blockedIP) {
//         return res.status(403).json({ message: "Access forbidden" });
//     }
//     next();
// };

// // IPブロックミドルウェアを適用
// app.use(blockIPMiddleware);

// connection pool
const pool = mysql.createPool({
    connectionLimit: 10,
    host: "localhost",
    port: 3307,
    user: "root",
    password: "",
    database: "test",
})

app.post("/sendData", (req, res) => {
    console.log(req.body.data);

    // MYSQLにcommentを保存
    pool.getConnection((err, connection) => {
        connection.query(`INSERT INTO comment values ("2", "${req.body.data}")`, (err, rows) => {
            connection.release();
            if (!err) {
                res.json({message: 'send data'});
            } else {
                console.log(err);
            }
        });
    });
});

app.get("/getData", (req, res) => {
    // 通信してきた相手のpiアドレスを取得
    pool.getConnection((err, connection) => {
        if (err) throw err;
        console.log('MYSQLと接続中・・・🌲');

        // データ取得
        connection.query("SELECT * FROM comment", (err, rows) => {
            connection.release();
            console.log(rows[0].text);
            if (!err) {
                res.json({message: rows[0].text});
            }
        })
    });
})

app.listen(PORT, () => {
    const host = ip.address();
    console.log('サーバー起動中🚀');
    console.log(`ip address is ${host}`)
});
