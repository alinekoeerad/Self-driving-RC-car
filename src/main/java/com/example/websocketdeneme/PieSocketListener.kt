package com.example.websocketdeneme

import android.util.Log
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener

class PieSocketListener : WebSocketListener() {
    override fun onOpen(webSocket: WebSocket, response: Response) {
        webSocket.send("")
        Log.e("burak","baglandi")
    }

    override fun onMessage(webSocket: WebSocket, text: String) {
        output("Received : $text")
    }

    override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
        webSocket.close(NORMAL_CLOSURE_STATUS, null)
        output("Closing : $code / $reason")
    }


    override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
        output("Error : " + t.message+"fsda")
    }

    private fun output(text: String?) {
        Log.d("PieSocket", text!!)
    }

    companion object {
        private const val NORMAL_CLOSURE_STATUS = 1000
    }
}