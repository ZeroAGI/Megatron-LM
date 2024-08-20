FROM nvcr.io/nvidia/pytorch:24.07-py3

ENV MPI_HOME=/usr/local/mpi
ENV NVTE_FRAMEWORK=pytorch
ENV NVTE_WITH_USERBUFFERS=1

RUN apt-get update && apt-get install net-tools -y

RUN pip install --no-cache-dir git+https://github.com/NVIDIA/TransformerEngine.git@bfe21c3d68b0a9951e5716fb520045db53419c5e

RUN pip install protobuf==3.20.*

RUN pip uninstall -y onnx && pip install --no-cache-dir onnx==1.16.1

RUN apt-get update && apt-get install -y openssh-server && \
    echo "Port 32321" >> /etc/ssh/sshd_config && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PermitEmptyPasswords yes" >> /etc/ssh/sshd_config && \
    echo "    Port 32321" >> /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config && \
    ssh-keygen -t rsa -f ~/.ssh/id_rsa -P '' && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

RUN /etc/init.d/ssh start

EXPOSE 32321

#RUN echo "/etc/init.d/ssh start" >> ~/.bashrc

#RUN service ssh restart 

#COPY entrypoint.sh /workspace/entrypoint.sh

#CMD ["/workspace/entrypoint.sh"]

CMD ["/usr/sbin/sshd", "-D"]
#CMD ["service", "ssh", "start"]

#ENTRYPOINT bash


