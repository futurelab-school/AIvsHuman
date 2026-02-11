# AIvsHuman
This is for the color mixing game

# Rules
### How to Play
- Adjust the CMYK sliders to match the target color.
- Hit the submit button when you want to make a guess.
- Keep trying until you get under 5% error.
- Try to beat the AI before it matches the color.
- You are given 8 random colors that the AI is also given for reference.


### Rules
1. Don't use the internet
2. You get one try per color. (Don't re-run the cell or you will lose your attemps)
3. Don't change the code!

```
Run the cell below to get started!

(\_/)                (\_/)
( •_•)              (•_• )
/>  /> ~ Have fun ! <\  <\

```


## To run it on google colab
``` # Get the game from the repo
  [python] !pip install git+https://github.com/futurelab-school/AIvsHuman.git

  from AIvsHuman import Color_Game
  # Launch the Game
  Color_Game().launch_game()
