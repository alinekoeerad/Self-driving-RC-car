# communication/csharp_communicator.py

import socket
import threading
import json
import struct
import time
from typing import Callable

class CSharpCommunicator:
    """
    Handles persistent, length-prefixed communication with the C# client
    by managing two separate, reconnecting TCP sockets.
    """

    def __init__(self, host: str, data_port: int, command_port: int):
        """Initializes the communicator."""
        self.host = host
        self.data_port = data_port
        self.command_port = command_port
        
        self.data_client_socket = None
        self.command_callback = None
        
        # A lock to prevent race conditions when multiple threads access the data socket
        self.socket_lock = threading.Lock()
        
        # An event to signal all threads to stop gracefully
        self.stop_event = threading.Event()
        
        print("📡 CSharpCommunicator initialized.")

    def register_command_callback(self, callback: Callable[[dict], None]):
        """Registers a function to be called when a command is received from C#."""
        self.command_callback = callback
        print("📞 Command callback registered.")

    def start(self):
        """Starts the data and command server threads."""
        self.stop_event.clear()
        
        # Start the data server in a separate daemon thread
        data_thread = threading.Thread(target=self._handle_persistent_connection, 
                                       args=(self.data_port, "Data", self._data_connection_handler), 
                                       daemon=True)
        data_thread.start()
        
        # Start the command server in another daemon thread
        command_thread = threading.Thread(target=self._handle_persistent_connection, 
                                          args=(self.command_port, "Command", self._command_connection_handler), 
                                          daemon=True)
        command_thread.start()
        print("🚀 C# communication threads started.")

    def stop(self):
        """Signals all threads to stop and closes sockets."""
        print("🔌 Shutting down C# communicator...")
        self.stop_event.set()
        
        # Close the socket to unblock any waiting operations
        with self.socket_lock:
            if self.data_client_socket:
                self.data_client_socket.close()

    def send_update(self, data_packet: dict):
        """
        Serializes a data packet to JSON and sends it to the C# client
        using the length-prefix protocol. This method is thread-safe.
        """
        with self.socket_lock:
            if self.data_client_socket:
                try:
                    # 1. Serialize the dictionary to a JSON string, then encode to bytes
                    message_bytes = json.dumps(data_packet).encode('utf-8')
                    
                    # 2. Pack the length of the message into a 4-byte, big-endian integer
                    length_prefix = struct.pack('!I', len(message_bytes))
                    
                    # 3. Send the 4-byte length prefix followed by the message bytes
                    self.data_client_socket.sendall(length_prefix + message_bytes)
                    
                except (BrokenPipeError, ConnectionResetError) as e:
                    print(f"❌ C# data client disconnected: {e}")
                    self.data_client_socket.close()
                    self.data_client_socket = None
                except Exception as e:
                    print(f"An unexpected error occurred during send_update: {e}")

    def _handle_persistent_connection(self, port: int, name: str, connection_handler_func: Callable):
        """
        A generic loop that listens for a single, persistent client.
        If the client disconnects, it goes back to listening for a new one.
        """
        while not self.stop_event.is_set():
            try:
                # Create a listening server socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    server_socket.bind((self.host, port))
                    server_socket.listen(1)
                    print(f"🎧 C# {name} server listening on {self.host}:{port}")
                    
                    # Accept a connection (this is a blocking call)
                    conn, addr = server_socket.accept()
                    print(f"✅ C# {name} client connected from {addr}")
                    
                    # Pass the connection to the specific handler function
                    connection_handler_func(conn)

            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"❌ Socket error on {name} server: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
        
        print(f"🔌 C# {name} server has shut down.")

    def _data_connection_handler(self, connection: socket.socket):
        """Handles the data sending socket. It's kept open for sending updates."""
        with self.socket_lock:
            self.data_client_socket = connection
            
        while not self.stop_event.is_set():
            try:
                # Use a lock to safely check and use the socket
                with self.socket_lock:
                    if self.data_client_socket:
                        # Send a small heartbeat packet to check the connection.
                        self.data_client_socket.sendall(struct.pack('!I', 0))
                    else:
                        # If socket is None, it was closed by another thread. Exit.
                        break
                
                time.sleep(2)
            except (socket.error, BrokenPipeError, ConnectionResetError):
                print("❌ Data client seems to have disconnected.")
                break # Exit the loop to allow for a new connection
        
        # Cleanup routine
        with self.socket_lock:
            if self.data_client_socket:
                self.data_client_socket.close()
                self.data_client_socket = None


    def _command_connection_handler(self, connection: socket.socket):
        """Handles the command receiving socket, reading length-prefixed messages."""
        with connection:
            while not self.stop_event.is_set():
                try:
                    # 1. Read the 4-byte length prefix. This is a blocking call.
                    length_prefix_bytes = connection.recv(4)
                    if not length_prefix_bytes:
                        print("❌ Command client disconnected gracefully.")
                        break

                    # 2. Unpack the 4 bytes into an integer
                    msg_len = struct.unpack('!I', length_prefix_bytes)[0]

                    # 3. Read the exact length of the message body
                    message_body = b''
                    while len(message_body) < msg_len:
                        chunk = connection.recv(msg_len - len(message_body))
                        if not chunk:
                            raise ConnectionError("Client disconnected during message receive.")
                        message_body += chunk
                    
                    # 4. Decode from bytes, parse JSON, and call the callback
                    if self.command_callback:
                        command_info = json.loads(message_body.decode('utf-8'))
                        self.command_callback(command_info)

                except (ConnectionResetError, ConnectionError) as e:
                    print(f"❌ C# command client disconnected: {e}")
                    break
                except Exception as e:
                    print(f"An unexpected error occurred in command receiver: {e}")
                    break