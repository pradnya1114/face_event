import sqlite3
import pickle

DB_PATH = "pre_registered/database.db"  # adjust path if needed

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Fetch all users
c.execute("SELECT id, name, face_encoding FROM users")
users = c.fetchall()

deleted_count = 0

for user in users:
    user_id, name, face_enc_blob = user
    if not face_enc_blob:
        # Empty encoding, delete
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        deleted_count += 1
        continue
    try:
        pickle.loads(face_enc_blob)
    except Exception as e:
        # Corrupted encoding, delete
        print(f"Deleting corrupted face encoding for {name}")
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        deleted_count += 1

conn.commit()
conn.close()

print(f"Database cleanup done. {deleted_count} invalid entries removed.")
