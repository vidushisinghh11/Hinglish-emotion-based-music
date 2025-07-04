import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

def get_spotify_client():
    client_id = "4f7699188ab742bdb8d5c0c85b64dd5d"
    client_secret = "4f7a09ccb2be4e8cb6fc5751db679aa4"
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp
