from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from pyActionRec.action_classifier import ActionClassifier
from pyActionRec.anet_db import ANetDB
import numpy as np
import youtube_dl
import urlparse
import tornado.wsgi
import tornado.httpserver

app=Flask(__name__)

UPLOAD_FOLDER='/data4/jiali/acTDD/tmp/'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

root='/data4/jiali/acTDD'
os.chdir(root)

#model specifications
models=[
    (root+'/models/resnet200_anet_2016_deploy.prototxt',
     root+'/models/resnet200_anet_2016.caffemodel',
     1.0,0,True,224),
    (root+'/models/bn_inception_anet_2016_temporal_deploy.prototxt',
     root+'/models/bn_inception_anet_2016_temporal_caffemodel.v5',
     0.2,1,False,224)
]

GPU=1

# init global variables
cls=ActionClassifier(models,dev_id=GPU)
db=ANetDB.get_db('1.3')
lb_list=db.get_ordered_label_list()

ydl=youtube_dl.YoutubeDL({u'outtmpl':u'tmp/%(id)s.%(ext)s'})


def allowed_file(filename):
    return



@app.route('/')
def main():
    return render_template('index.html')


def build_cls_ret(scores,k):
    idx=np.argsort(scores)[::-1]

    top_k_results=[]
    for i in xrange(k):
        k=idx[i]
        top_k_results.append({
            'name':lb_list[k],
            'score':str(scores[k])
        })
    return top_k_results


def run_classification(filename,use_rgb,use_flow):
    scores,frm_scores,total_time=cls.classify(filename,[use_rgb=='true',use_flow=='true'])

    ret=build_cls_ret(scores,3)

    return jsonify(error=None,results=ret,total_time=total_time,n_snippet=len(frm_scores),fps=1),200,{'ContentType':'application/json'}





@app.route('//uplaod_video',methods=['POST'])
def upload_video():
    if 'video_file' not in request.files:
        return jsonify(error='Upload not found'),200, {'ContentType':'application/json'}

    upload_file=request.files['video_file']
    if upload_file.filename=='':
        return jsonify(error='file has no name'),200,{'ContentType':'application/json'}

    use_rgb=request.form['use_rgb']
    use_flow=request.form['use_flow']

    if upload_file and allowed_file(upload_file.filename):
        filename =secure_filename(upload_file.filename)

        # first save the file
        savename=os.path.join(app.config['UPLOAD_FOLDER'],filename)
        print 'savename: %s' % savename
        upload_file.save(savename)


        # classify the video
        return run_classification(savename,use_rgb=use_rgb,use_flow=use_flow)
    else:
        return jsonify(error='empty or not allowed file'),200,{'ContentType':'application/json'}


def start_tornado(_app,port=5000):
    http_server=tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(_app)
    )
    http_server.listen(port)
    print('Tornado server starting on port {}'.format(port))
    tornado.ioloop.IOLoop.instance().start()


if __name__=='__main__':
    start_tornado(app,5903)
    
