import json
 
def generate_subtitles(text, video_length):
    # Clean text and split at periods and commas
    text = ' '.join(text.split())
    splits = [s.strip() for s in text.replace('.', ',').split(',') if s.strip()]
 
    # Calculate timing
    content_duration = video_length - 4
    segment_duration = content_duration / len(splits)
 
    subtitles = [{"index": 0, "start": "0.00", "end": "2.00", "text": ""}]
 
    current_time = 2.0
    for i, segment in enumerate(splits, 1):
        end_time = current_time + segment_duration
        subtitles.append({
            "index": i,
            "start": f"{current_time:.2f}",
            "end": f"{end_time:.2f}",
            "text": segment
        })
        current_time = end_time
 
    subtitles.append({
        "index": len(subtitles),
        "start": f"{current_time:.2f}",
        "end": f"{current_time + 2:.2f}",
        "text": ""
    })
 
    return {"subs": subtitles}
 