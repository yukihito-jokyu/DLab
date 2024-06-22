import { io } from "socket.io-client";

const socket = io('ws://127.0.0.1:5000');

export { socket };