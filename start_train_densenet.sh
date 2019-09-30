pid=`ps -ef|grep "python -u tools/train_densenetocr_ctc_multigpu.py"| grep -v "grep"|awk '{print $2}'`

if [ "$pid" != "" ]
then
	        echo "tools/train_densenetocr_ctc_multigpu.py already run, stop it first"
		        kill -9 ${pid}
		fi

		echo "starting now..."

		nohup python -u tools/train_densenetocr_ctc_multigpu.py > train_densenet.out 2>&1 &

		pid=`ps -ef|grep "python -u tools/train_densenetocr_ctc_multigpu.py"| grep -v "grep"|awk '{print $2}'`

		echo ${pid} > train_densenet.out
		echo "tools/train_densenetocr_ctc_multigpu.py started at pid: "${pid}

