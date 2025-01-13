from pymongo import MongoClient
from dotenv import load_dotenv
import os

def test_mongodb_connection():
    # Load environment variables
    load_dotenv()
    
    # Get MongoDB connection details from environment variables
    uri = os.getenv('MONGO_URI')
    db_name = os.getenv('MONGO_DB_NAME')
    collection_name = os.getenv('MONGO_COLLECTION_NAME')
    
    print(f"Attempting to connect with:")
    print(f"Database: {db_name}")
    print(f"Collection: {collection_name}")
    
    try:
        # Create a MongoDB client
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        
        # Test the connection
        client.server_info()
        print("✅ Successfully connected to MongoDB server")
        
        # Get database and collection
        db = client[db_name]
        collection = db[collection_name]
        
        # Test if we can fetch one document
        doc = collection.find_one()
        if doc:
            print("✅ Successfully retrieved a document")
            print("\nDocument preview:")
            print("- Keys available:", list(doc.keys()))
            if 'kps_2d' in doc:
                print("- Number of 2D keypoints:", len(doc['kps_2d']))
            if 'kps_3d' in doc:
                print("- Number of 3D keypoints:", len(doc['kps_3d']))
        else:
            print("⚠️ Collection is empty")
            
    except Exception as e:
        print("❌ Failed to connect to MongoDB:")
        print(str(e))
        print("\nPlease check:")
        print("1. MongoDB server is running")
        print("2. Your .env file has correct values:")
        print("   MONGO_URI=<your_uri>")
        print("   MONGO_DB_NAME=<your_database>")
        print("   MONGO_COLLECTION_NAME=<your_collection>")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    test_mongodb_connection() 
