import asyncio
import websockets
import json

async def send_message(uri, message):
    """Sends a message to the WebSocket server and prints the response."""
    try:
        async with websockets.connect(uri) as websocket:
            print(f"> Sending to {uri}: {message}")
            await websocket.send(message)

            response = await websocket.recv()
            print(f"< Received from {uri}: {response}")
            return response
    except ConnectionRefusedError:
        print(f"Connection to {uri} refused. Ensure the server is running.")
    except Exception as e:
        print(f"An error occurred: {e}")

async def main():
    # RAG App 2 (Client App with forwarding logic) runs on port 8002
    client_app_uri = "ws://localhost:8002/ws/chat"

    print("-" * 30)
    print("Testing RAG App 2 (Client with Forwarding Logic)")
    print("-" * 30)

    # 1. Test local processing on RAG App 2
    print("\n--- Test 1: Query for Local Processing (e.g., local weather) ---")
    # This query should be handled by RAG App 2 itself using its local documents
    # (e.g., "This is a document from the Client RAG App. It focuses on local weather patterns...")
    local_query = "Tell me about local weather patterns"
    await send_message(client_app_uri, local_query)

    # 2. Test forwarding to MCP Server (RAG App 1) via RAG App 2
    print("\n--- Test 2: Query for Forwarding to MCP Server (e.g., quantum computing) ---")
    # This query includes the keyword "FORWARD_TO_MCP" and should be forwarded by RAG App 2
    # to RAG App 1 (MCP Server), which has documents about quantum computing.
    forward_query = "FORWARD_TO_MCP: Tell me about advanced quantum computing protocols"
    await send_message(client_app_uri, forward_query)

    # 3. Test another local query to ensure session continuity
    print("\n--- Test 3: Another Local Query ---")
    local_query_2 = "Any tips for regional gardening?"
    await send_message(client_app_uri, local_query_2)

    # 4. Test a query that might not be in either, to see default response or error
    print("\n--- Test 4: Query not in local or MCP (example) ---")
    # This query is unlikely to be in the sample documents.
    # If not forwarded, App 2 handles it. If forwarded, App 1 handles it.
    # The response will depend on the LLM's handling of out-of-context queries.
    unknown_query_local = "What is the airspeed velocity of an unladen swallow?"
    await send_message(client_app_uri, unknown_query_local)

    print("\n--- Test 5: Forwarded query for something not in MCP docs (example) ---")
    unknown_query_forwarded = "FORWARD_TO_MCP: What is the best recipe for apple pie?"
    await send_message(client_app_uri, unknown_query_forwarded)

    print("\n" + "-" * 30)
    print("Test script finished.")
    print("Ensure RAG App 1 (MCP Server) is running on port 8001 and RAG App 2 (Client) is running on port 8002.")
    print("Also ensure documents have been processed for both apps.")
    print("-" * 30)

if __name__ == "__main__":
    asyncio.run(main())
