pid=`ps -ef|grep "python -u tools/train_mobileNetV2_ctc_multigpu.py"| grep -v "grep"|awk '{print $2}'`

if [ "$pid" != "" ]
then
	        echo "tools/train_mobileNetV2_ctc_multigpu.py already run, stop it first"
		        kill -9 ${pid}
		fi

		echo "starting now..."

		nohup python -u tools/train_mobileNetV2_ctc_multigpu.py > train_mobilenetV2.out 2>&1 &

		pid=`ps -ef|grep "python -u tools/train_mobileNetV2_ctc_multigpu.py"| grep -v "grep"|awk '{print $2}'`

		echo ${pid} > pid.out
		echo "tools/train_mobileNetV2_ctc_multigpu.py started at pid: "${pid}

