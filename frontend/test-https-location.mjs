import assert from "node:assert/strict";
import { muxLooksLikeHttp, livekitConnectUrl } from "./https-location.mjs";

assert.equal(muxLooksLikeHttp(Buffer.from("GET / HTTP/1.1\r\n")), true);
assert.equal(muxLooksLikeHttp(Buffer.from("POST /api HTTP/1.1\r\n")), true);
assert.equal(muxLooksLikeHttp(Buffer.from([0x16, 0x03, 0x03, 0x00])), false);
assert.equal(muxLooksLikeHttp(Buffer.from([0x17, 0x03, 0x03, 0x00])), false);
assert.equal(muxLooksLikeHttp(Buffer.from([0x14, 0x03, 0x03])), false);
assert.equal(muxLooksLikeHttp(Buffer.from([0x15, 0x03, 0x03])), false);
assert.equal(muxLooksLikeHttp(Buffer.from([0x80, 0x00])), false);
assert.equal(muxLooksLikeHttp(Buffer.from("HELLO")), false);

const page = { protocol: "https:", host: "10.4.7.193:6173", hostname: "10.4.7.193" };
assert.equal(livekitConnectUrl(page, "wss://10.4.7.193:6173"), "wss://10.4.7.193:6173");
assert.equal(livekitConnectUrl(page, "ws://127.0.0.1:8880"), "wss://10.4.7.193:6173");

console.log("https-location ok");
