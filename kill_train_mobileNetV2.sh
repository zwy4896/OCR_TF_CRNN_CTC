pid=`ps -ef|grep "python -u tools/train_mobileNetV2_ctc_multigpu.py"| grep -v "grep"|awk '{print $2}'`

if [ "$pid" != "" ]
then
	        kill -9 ${pid}
		        echo "stop tools/train_mobileNetV2_ctc_multigpu.py complete"
		else
			        echo "tools/train_mobileNetV2_ctc_multigpu.py is not run, there's no need to stop it"
			fi

