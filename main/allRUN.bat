@echo on
call activate
call conda activate {your environment}
python run_pre.py --key=LOG4NET --pro=logging-log4net
python run_pre.py --key=GIRAPH --pro=Giraph
python run_pre.py --key=NUTCH --pro=Nutch
python run_pre.py --key=OODT --pro=OODT
python run_pre.py --key=KERAS --pro=keras