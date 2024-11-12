import sqlite3
from datetime import datetime

# pip install these
import vlc
import yt_dlp
import pygame

"""
This is a very simple implementation of an attendance db which has a lot of flaws.
Currently the database isn't even being used just stored, there's lots of potential here.
"""
def init_db():
    """Initialize the attendance database."""

    # Connect to the SQLite database (it will create the database if it doesn't exist)
    conn = sqlite3.connect('attendance.db') 
    cursor = conn.cursor()

    # Create a table 'attendance' with columns for name and timestamp if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance
                   (
                   name TEXT, 
                   streak INTEGER DEFAULT 0,
                   total_attendance INTEGER DEFAULT 0,
                   recent_attendance_date TEXT
                   )''')
    
    # create a table 'music' to store student's music details (youtube or local file)
    cursor.execute('''CREATE TABLE IF NOT EXISTS music
                   (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   name TEXT,
                   song_title TEXT,
                   artist TEXT,
                   song_path TEXT, -- Local File path
                   youtube_url TEXT, -- Youtube URL
                   FOREIGN KEY(name) REFERENCES attendance(name)
                   )''')
    

    # Commit changes and close the connection
    conn.commit()
    conn.close()

# after attendance is marked, we play their song
def mark_attendance(name):

    """Insert a new attendance record into the database."""
    # Connect to the SQLite database
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    # Stores today's date
    today = datetime.now().strftime('%Y-%m-%d')

    # Checks's if student exist in db
    cursor.execute("SELECT streak, total_attendance, recent_attendance_date FROM attendance WHERE name = ?", (name,))
    record = cursor.fetchone()

    # if student alr exist..
    if record:
        streak, total_attendance, recent_attendance_date = record
        
        # Calcualtes days since last meeting (should be 7)
        days_since = (datetime.now() - recent_attendance_date).days

        # checks if it is a streak
        if days_since == 7:
            streak += 1
        else:
            # breaks streak
            streak = 1

        # adds to total attendance
        total_attendance += 1

        # Update the record
        cursor.execute(''' UPDATE attendance 
        SET streak = ?, 
        total_attendance = ?,
        recent_attendance_date = ? 
        WHERE name = ?
        ''', (streak, total_attendance, today, name))

    # add student to db if does not exist alr
    else:
        cursor.execute('''INSERT INTO attendance 
        (name, streak, 
        total_attendance,
        recent_attendance_date) 
        VALUES (?, 1, 1, ?)
        ''', (name, today))


    # play song
    play_student_song(name)



    # Commit changes and close the connection
    conn.commit()
    conn.close()


# this works for audio files stored locally
# defaults path and url to none
def add_music(name, song_name, artist, song_path=None, youtube_url=None):
    """Add a music entry for a student with either a local file path or a YouTube URL."""


    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()


    # see if student has a song already
    cursor.execute("SELECT id FROM music WHERE name = ?", (name,))
    result = cursor.fetchone()

    # if there is a song stored for them...
    if result:

        # Update the existing record with the new information
        cursor.execute('''UPDATE music 
                          SET song_name = ?, 
                              artist = ?, 
                              song_path = ?, 
                              youtube_url = ? 
                          WHERE name = ?''', 
                       (song_name, artist, song_path, youtube_url, name))
        
    else:

        # Insert a new record if the student doesn't exist

        cursor.execute('''INSERT INTO music (name, song_name, artist, song_path, youtube_url)
                          VALUES (?, ?, ?, ?, ?)''', 
                       (name, song_name, artist, song_path, youtube_url))

    conn.commit()
    conn.close()



    '''
    example use:
    init_db()
    mark_attendance('Chris')
    add_music('Chris', 'On The Floor', 'Jennifer Lopez', '/path/to/song.mp3')
    '''

# function: plays song from either youtube or local file (MP3)
# if song is not found for the input, then we play nothing
def play_student_song(name):

    """Play the student's song, either from a local file or YouTube URL."""

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    # gets information for song
    cursor.execute("SELECT song_path, youtube_url FROM music WHERE name = ?", (name,))
    result = cursor.fetchone()

    if result:

        song_path, youtube_url = result

        if song_path:

            # plays file song with pygame
            pygame.mixer.init()
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
        
        elif youtube_url:

            # plays youtube song with yt_dlp & ydl
            ydl_opts = {'format': 'bestaudio'}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                stream_url = info['url']

            player = vlc.MediaPlayer(stream_url)
            player.play()


    # if song cannot be found
    else:
        print(f"No music found for {name}")

    conn.close()
    
