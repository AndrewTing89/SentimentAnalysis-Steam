# --- Your task.py file, in the train_production_model function ---

    # 4. Save and Upload Model to GCS
    print("\n💾 Saving and uploading model...")
    stamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    local_file_name = "model.joblib"
    
    # --- DEBUGGING STEP 1: Save model locally and check existence ---
    print(f"DEBUG: Attempting to save model locally to: {local_file_name}")
    try:
        joblib.dump(prod_pipe, local_file_name)
        print(f"DEBUG: Model saved locally. Does '{local_file_name}' exist? {os.path.exists(local_file_name)}")
    except Exception as e:
        print(f"ERROR: Failed to save model locally. Exception: {type(e).__name__}: {e}")
        raise # Re-raise to stop the job if local save fails

    # --- DEBUGGING STEP 2: Gzip model locally and check existence ---
    gzipped_local_file_name = f"{local_file_name}.gz"
    print(f"DEBUG: Attempting to gzip model to: {gzipped_local_file_name}")
    try:
        with open(local_file_name, "rb") as fin, gzip.open(gzipped_local_file_name, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        print(f"DEBUG: Model gzipped. Does '{gzipped_local_file_name}' exist? {os.path.exists(gzipped_local_file_name)}")
    except Exception as e:
        print(f"ERROR: Failed to gzip model locally. Exception: {type(e).__name__}: {e}")
        raise # Re-raise to stop the job if gzip fails

    # --- DEBUGGING STEP 3: Prepare GCS path and attempt upload ---
    model_directory = f"models/lr-tfidf/{stamp}"
    storage_path = os.path.join(model_directory, gzipped_local_file_name)
    
    print(f"DEBUG: Preparing to upload to GCS.")
    print(f"DEBUG: Target Bucket: {bucket_name}")
    print(f"DEBUG: Target Blob Path: {storage_path}")

    try:
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(storage_path)
        
        # Final check if the gzipped file exists just before upload
        if not os.path.exists(gzipped_local_file_name):
            raise FileNotFoundError(f"Local gzipped model file '{gzipped_local_file_name}' not found before upload.")

        print(f"DEBUG: Starting upload of '{gzipped_local_file_name}' to GCS.")
        blob.upload_from_filename(gzipped_local_file_name) # Ensure this uses the gzipped file name
        
        print(f"✅ Model uploaded to: gs://{bucket_name}/{storage_path}")
    except Exception as e:
        # --- CRITICAL: This will print the actual error type and message! ---
        print(f"\nFATAL ERROR: Failed to upload model to GCS.")
        print(f"Exception Type: {type(e).__name__}")
        print(f"Exception Message: {e}")
        raise # Re-raise the exception to ensure the job fails clearly