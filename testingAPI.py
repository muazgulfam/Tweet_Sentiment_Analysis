import tweepy

# Replace these with your actual keys
API_KEY = "KhS4Qz98uHiS4VSelyEQqYuFd"
API_SECRET = "U8XkrNXz68RzWeMKZIe54VSUn4ijxonAUooc92LmF5wlkYSAko"
ACCESS_TOKEN = "1388936886438203394-gCvDFKwWIkJ3SgqhoCxKG6ed4GGNNK"
ACCESS_TOKEN_SECRET = "CObWofrpjQWJVVElJ9iuKrub8gTMP0G1L79huuMtTr09w"

# Authenticate
auth = tweepy.OAuth1UserHandler(
    API_KEY, 
    API_SECRET, 
    ACCESS_TOKEN, 
    ACCESS_TOKEN_SECRET
)

# Verify login
api = tweepy.API(auth)
try:
    api.verify_credentials()
    print("✅ Twitter API Connected!")
except Exception as e:
    print("❌ Error:", e)