
From centos

RUN yum -y update && yum install -y git wget python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install  numpy==1.15.4 
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple ipython
RUN pip3 install  scikit-learn
RUN pip3 install  scikit-image
RUN pip3 install  jinja2
RUN pip3 install  imageio
RUN pip3 install  h5py==2.10.0 
RUN pip3 install  pyyaml
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple keras==2.1.0
RUN pip3 install  tensorflow==1.12.0
RUN pip3 install  pyxb
RUN pip3 install  lxml
RUN pip3 install  jsonschema
RUN pip3 install  xmljson
RUN pip3 install  memory-profiler
RUN mkdir AI
COPY AI  /AI
