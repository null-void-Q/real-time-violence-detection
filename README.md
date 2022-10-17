# real-time-violence-detection
real-time violence detection on video content.

This is a demo of a violence detection deep learning model. The model is based on Deepminds [kinetics-i3d](https://github.com/deepmind/kinetics-i3d "kinetics-i3d") and was developed using transfer learning.
The model was trained on a dataset of 35512 video clips (64 frame/clip) and produced the following results on the test data:

Accuracy       | Precision | Recall
-------------- | :-----------------: | -----------
91.6%        | 89.58%        | 84.71%

## Running the code
the code was tested on an nvidia GPU with cudnn drivers, it might not run properly on cpu .
####  Commands
clone the repository: <br/>
`git clone https://github.com/null-void-Q/real-time-violence-detection.git`
<br/>
install requirments: <br/>
`pip install -r requirments.txt` 
<br/>
run the demo: <br/>
`python main.py` 
<br/>

after the local server is up head to [localhost:5000](localhost:5000 "localhost:5000") in your browser to use the web interface. (included test_video.mp4)


[![INterface](https://user-images.githubusercontent.com/53970206/196257994-be5d8a86-3365-4d87-b5c7-dd7a1eeb8122.png "INterface")](http://https://user-images.githubusercontent.com/53970206/196257994-be5d8a86-3365-4d87-b5c7-dd7a1eeb8122.png "INterface")

<br/><br/>


[![Interface](https://user-images.githubusercontent.com/53970206/196257986-a392c2bc-8764-4747-83f0-28838d232a6b.png "Interface")](https://user-images.githubusercontent.com/53970206/196257986-a392c2bc-8764-4747-83f0-28838d232a6b.png "Interface")

