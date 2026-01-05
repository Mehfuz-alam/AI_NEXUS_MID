from authlib.integrations.starlette_client import OAuth
import os
oauth = OAuth()

oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),       # <-- must read env
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),  # <-- must read env
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# oauth.register(
#     name="facebook",
#     client_id="FACEBOOK_APP_ID",
#     client_secret="FACEBOOK_APP_SECRET",
#     authorize_url="https://www.facebook.com/v12.0/dialog/oauth",
#     access_token_url="https://graph.facebook.com/v12.0/oauth/access_token",
#     api_base_url="https://graph.facebook.com/",
#     client_kwargs={"scope": "email"},
# )
