echo '[killing previous threads]'
pkill -f elasticsearch
sleep 2
# /tmp2/py/data/fever/elasticsearch-7.13.2/bin/elasticsearch -d -p pid -E http.port=1234
/tmp2/py/data/fever/elasticsearch-7.13.2/bin/elasticsearch -p pid -E http.port=1234