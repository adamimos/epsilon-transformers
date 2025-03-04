import boto3
import os
from dotenv import load_dotenv
from tqdm.auto import tqdm
import subprocess

# Load environment variables
load_dotenv()

# Bucket names
SOURCE_BUCKET = "quantum-runs"
DEST_BUCKET = os.getenv('COMPANY_S3_BUCKET_NAME')

print(f"Source bucket: {SOURCE_BUCKET}")
print(f"Destination bucket: {DEST_BUCKET}")

if not DEST_BUCKET:
    raise ValueError("Missing environment variable: COMPANY_S3_BUCKET_NAME")

# For destination bucket - use the default profile that works
dest_session = boto3.Session(profile_name='default')
dest_s3 = dest_session.client('s3')

# For source bucket - use explicit credentials from environment variables
source_s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
)

# Verify destination bucket access
print("\nTesting destination bucket access...")
try:
    test_key = "test_permission_delete_me"
    dest_s3.put_object(Bucket=DEST_BUCKET, Key=test_key, Body="test")
    print(f"✅ Destination bucket write access OK")
    dest_s3.delete_object(Bucket=DEST_BUCKET, Key=test_key)
    print(f"✅ Destination bucket delete access OK")
except Exception as e:
    print(f"❌ Cannot write to destination bucket: {e}")
    raise ValueError("No access to destination bucket")

# Verify source bucket access
print("\nTesting source bucket access...")
try:
    response = source_s3.list_objects_v2(Bucket=SOURCE_BUCKET, MaxKeys=1)
    if 'Contents' in response:
        print(f"✅ Source bucket access OK. Found objects.")
    else:
        print(f"✅ Source bucket access OK. No objects found.")
except Exception as e:
    print(f"❌ Cannot access source bucket: {e}")
    print("\nPlease check your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
    print("These should be your personal credentials that have access to the quantum-runs bucket.")
    raise ValueError("No access to source bucket")

# Sweeps to migrate
sweeps_to_migrate = ['20241205175736', '20241121152808']

def migrate_sweep(sweep_id):
    """Migrate all files for a given sweep"""
    print(f"\nMigrating sweep: {sweep_id}")
    
    # List all objects in the sweep using source credentials
    objects = []
    paginator = source_s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=SOURCE_BUCKET, Prefix=f"{sweep_id}/"):
        if 'Contents' in page:
            objects.extend(page['Contents'])
    
    print(f"Found {len(objects)} objects to migrate")
    
    # Copy each object
    for obj in tqdm(objects, desc="Copying files"):
        source_key = obj['Key']
        dest_key = f"quantum_runs/{source_key}"
        
        try:
            # Create a temporary file path
            tmp_file = f"/tmp/{os.path.basename(source_key)}"
            
            # Download from source using source credentials
            source_s3.download_file(SOURCE_BUCKET, source_key, tmp_file)
            
            # Upload to destination using destination credentials
            dest_s3.upload_file(tmp_file, DEST_BUCKET, dest_key)
            
            # Clean up
            os.remove(tmp_file)
        except Exception as e:
            print(f"Error copying {source_key}: {e}")

def main():
    """Main migration function"""
    for sweep_id in sweeps_to_migrate:
        migrate_sweep(sweep_id)
    print("\nMigration complete!")

if __name__ == "__main__":
    main()