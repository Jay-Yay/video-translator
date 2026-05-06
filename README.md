# video-translator

## How to Set Up
1. pip install -r requirements.txt
2. Create an .env file
```
ANTHROPIC_API_KEY="{PutYourKey}"
ELEVENLABS_API_KEY="{PutYourKey}"
ELEVENLABS_VOICE_ID_MALE="zdWu2I1sJrZsmxJG5rie"
ELEVENLABS_VOICE_ID_FEMALE="fUjY9K2nAIwlALOwSiwc"
```
3. Issue an API key from https://console.cloud.google.com/apis/dashboard, download 'key.json' file, and add it to the root of this project.
4. Create these folder at the root
  - videos-to-be-done
  - videos-done
  - checkpoints
5. Put videos in the 'videos-to-be-done' folder



## Commands 
### Full pipeline (all stages):                                                                                                            
python process_translation_jp.py                                                                                                

                                                                                                                                       
### Skip Stage 1 — use existing detections.json:                                                                                           
python process_translation_jp.py --skip-detect                                                                                         

                                                                                                                                       
### Skip Stage 2 — use existing translations.json:                                                                                         
python process_translation_jp.py --skip-translate                                                                                      

                                                                                                                                       
### Force re-run all stages (ignore checkpoints):                                                                                          
python process_translation_jp.py --force                  


### Individual stage scripts (run directly if needed):
python stages/detect_gvi.py       # Stage 1: GVI text detection
python stages/translate.py        # Stage 2: Claude translation
python stages/tts.py              # Stage 3: TTS generation
python stages/export_prproj.py    # Stage 4: Premiere Pro project export


### Individual stage scripts (run directly if needed):
python stages/detect_gvi.py       # Stage 1: GVI text detection
python stages/translate.py        # Stage 2: Claude translation
python stages/tts.py              # Stage 3: TTS generation
python stages/export_prproj.py    # Stage 4: Premiere Pro project export


### TTS utilities:
python generate_tts_all.py        # Generate TTS for all videos
python generate_tts_one.py        # Generate TTS for a single video



## How to Use
1. Put videos to the folder `videos-to-be-done/` as .mp4 files
2. Run the script
3. Get outputs from the folder `videos-done/`
