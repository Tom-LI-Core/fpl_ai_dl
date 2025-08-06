# FPL AI – Deep-Learning Assistant

A Python package that:

* pulls data from the official FPL API + Understat + injury feeds  
* builds a weekly player-level dataset (`player_weeks.parquet`)  
* trains an LSTM to predict expected points  
* recommends transfers, captain, bench order, and watches price moves.

> **Quick start (Colab)**  
> ```bash
> !git clone https://github.com/<Tom-LI-Core>/fpl_ai_dl.git
> %cd fpl_ai_dl
> !pip install -r requirements.txt
> !python -m fpl_ai.scripts.update_data
> !python -m fpl_ai.models.lstm
> !python -m fpl_ai.scripts.week_run --team_json my_team.json
> ```
