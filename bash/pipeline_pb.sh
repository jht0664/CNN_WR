# Pipeline from preprocessing to transition point
#  assume all input files have the same mole fraction
# v 0.1 - clean scripts for simplicity.

# User-defined variables
path_bash='/home/htjung/upload_github/CNN_WR/bash/' # path for bash scripts 
path_py='/home/htjung/upload_github/CNN_WR/'        # path for CNN_WR github directory
path_model='/home/htjung/upload_github/CNN_WR/model/' # path for model.config.* files
run_gmx='gmx'         # Gromacs command

n_copies_train=400    # ensembles for training set, 0 = min, -1 = max
n_copies_eval=10      # ensembles for evaluation set, 0 = min, -1 = max
temp1=1.0             # training temp1 or density1
temp2=4.0             # training temp2 or density2
crit=0.3              # valid transition temperature range
					  #  between temp1+crit and temp1-crit   
n_grids=13            # #grids on axes
n_files_train=1       # files for training data
n_files_eval=2        # files for evaluation data
proportion_test=0.0   # sample percentage for testing
start_model=0         # start index of model.config
end_model=100         # end index of model.config
repl_text="2 16"    # text optimal size and #filters for model.config

# main 
echo "select index of steps you want to start... [0-5]"
echo "1) make super cells from gro files and grid interpolation with copy ensembles"
echo "    *requires gmx (Gromacs) and bc command"
echo "    *activate mdtraj_env to read gro file"
echo "2) make blocks for train/test/eval sets"
echo "    *run this at head node with a large memory space > 32GB"
echo "3) make machine learning model with training sets"
echo "    *run this after activate tensorflow-gpu"
echo "4) run evaluation with previous ML models"
echo "    *run this after activate tensorflow-gpu"
echo "5) fit for Tc and make figures for candidate CNN models" 
echo "    *run this after activate matplotlib python module"
read var_step

case $var_step in
1 ) # step1: make super cells and grid interpolation into npy files
	if [ ! -d output ]; then
		echo "you do not have output folder with .gro files. Check"
		exit 0
	fi
	# rename gro file if necessary
	cd output
	do_or_not=$(ls confout.gro.* | wc | awk '{print $1}')
	if [ $do_or_not -ge 1 ]; then 
		i=0
		while [ $i -lt $do_or_not ]; do
			if [ ! -f "confout.gro."$i ]; then
				echo "no file of confout.gro."$i
			else
				mv "confout.gro."$i "confout."$i".gro"
			fi
			let i=$i+1
		done
	fi
	cd ..	
	# make target.list if available
	#  target.list format  : $file_index $temperature
	#  make_init.log format: #atoms      #A    #B   #temp   #i_file
	if [ ! -f output/make_init.log ]; then
		echo "you do no have make_init.log in machine folder"
		if [ -f output/target.list ]; then
                        echo "but you have target.list. continue..."
                else
                        echo "stop.."
                        exit 0
                fi
	else
		n_columns=$(head -1 output/make_init.log | sed 's/[^ ]//g' | wc -c)
		if [ $n_columns -eq  5 ]; then
			awk '{print $4 " " $5}' output/make_init.log > output/target.list
		elif [ $n_columns -eq 4 ]; then
			awk '{print $3 " " $4}' output/make_init.log > output/target.list
		else
			echo "something wrong on target.list file?"
			exit 0
		fi
	fi
	# make idx array for train set by reading target.list file
	if [ -d grid ]; then
		echo "already grid folder. skip to make folder"
	else
		mkdir grid
	fi
	cp output/target.list grid/
	file="grid/target.list" 
	declare -a trainset evalset
	while IFS=' ' read -r f1 f2; do 
		if (( $(echo "$f1 <= ${temp1} || $f1 >= ${temp2}" | bc -l) )); then 
			trainset+=($f2)
		else
			evalset+=($f2)
		fi
	done < $file
	# do preparation step
	
	cd grid
	mkdir train; mkdir eval
	# make 2x2x2 super cell & make npy file for grid interpolation
	for it in "${trainset[@]}"; do
		if [ -f "grid."$it".npy" ]; then
			continue
		fi
		# all atoms wrap within cell
		echo 0 | $run_gmx trjconv -f "../output/confout."$it".gro" -s "../output/confout."$it".gro" -pbc atom -o "train/confout."$it".gro"
		$run_gmx genconf -f "train/confout."$it".gro" -nbox 2 2 2 -o "train/super."$it".gro"
		python $path_py"grid/interpolate.py" -name PB -i "train/super."$it".gro" -g $n_grids -s 2 -itp three-states -n_ensembles $n_copies_train -o "grid."$it
	done
	for it in "${evalset[@]}"; do
		if [ -f "grid."$it".npy" ]; then
			continue
		fi
		# all atoms wrap within cell
		echo 0 | $run_gmx trjconv -f "../output/confout."$it".gro" -s "../output/confout."$it".gro" -pbc atom -o "eval/confout."$it".gro"
		$run_gmx genconf -f "eval/confout."$it".gro" -nbox 2 2 2 -o "eval/super."$it".gro"
		python $path_py"grid/interpolate.py" -name PB -i "eval/super."$it".gro" -g $n_grids -s 2 -itp three-states -n_ensembles $n_copies_eval -o "grid."$it
	done
	cd ..
	# returns grid.$i.npy files
	;;
2 ) # step2: make blocks for training set
	#   and calculate accuracy on some portion of training set (test set)
	#   and generate a single file for Tc evaluation data set
	#  for machine learning for Widom-Rowlinson model
	cd grid
	python $path_py"machine/block.py" -i target.list -ipf grid -s1 $temp1 -s2 $temp2 -prop $proportion_test -nb $n_files_train -seed 1985 -ng $n_grids -nbe $n_files_eval -net $n_copies_train -nee $n_copies_eval
	cd ..
	# returns train.(coord/temp/cat).$i.npy, test.(coord/temp/cat).npy, and eval.(coord/temp).npy
	;;
3 ) # step3: make machine learning model with some CNN models
	if [ ! -d machine ]; then
		if [ ! -d grid ]; then
			echo "no grid folder?"
			exit 0
		fi
		mkdir machine
	else
		if [ ! -f machine/train.0.coord.npy ]; then
			echo "no training set?"
			exit 0
		fi
	fi
	if [ -d grid ]; then
		mv grid/train*.npy machine/
		mv grid/eval*.npy  machine/
		mv grid/test*.npy  machine/
	fi
	cd machine
	echo "copy models into machine folder"
	echo "you should activate tensorflow-gpu like:"
	echo "source /share/apps/tensorflow-gpu/bin/activate"
	if [ ! -f model.config.0 ]; then
		cp ${path_model}* ./
		sed -i "s#4 16#${repl_text}#g" model.config.*
	fi
	# run machin learning
	i_model=$start_model
	while [ $i_model -lt $end_model ]; do
		#echo "... running " $i_model "th model ..."
		i=0
		config_file="model.config."$i_model
		if [ ! -f $config_file ]; then
			let i_model=$i_model+1
			continue
		fi
		while [ $i -lt $n_files_train ]; do
			input_file="train."$i
			out_model="model."$i_model"."$i".h5"
			if [ ! -f $out_model ]; then
			        python $path_py"machine/train.py" -nb 50 -i $input_file -it "test" -ng $n_grids -config $config_file -o $out_model
			else
			        echo "already model.h5 exists"
			        exit 0
			fi
			let i=$i+1
		done
		let i_model=$i_model+1
	done
	echo "Done"
	exit 0
	cd ..
	# returns "model.result."$i_model"."$i".npy" files
	;;
4 ) # step4: run evaluation with previous ML models
	cd machine
	# run prediction
	i_model=$start_model
	while [ $i_model -lt $end_model ]; do
		#echo "... running " $i_model "th model ..."
		i=0
		config_file="model.config."$i_model
		if [ ! -f $config_file ]; then
			let i_model=$i_model+1
			continue
		fi
		while [ $i -lt $n_files_train ]; do
			out_file="model.result."$i_model"."$i".npy"
			input_model="model."$i_model"."$i".h5"
			if [ ! -f $input_model ]; then
				echo "no input model " $input_model
				break
			fi
			#filter_size=$(sed -n '2p' $config_file | awk '{print $1}')
			#mf=$(bc -l <<< $filter_size*2)
			#if [ $mf -gt $n_grids ]; then
			#       break
			#fi
			python $path_py"machine/predict.py" -m $input_model -i eval -nf $n_files_eval -ne $n_copies_eval -ng $n_grids -o $out_file
			let i=$i+1
		done
		let i_model=$i_model+1
	done
    cd ..
	;;
5 ) # step5: fit for Tc and make figures for candidate CNN models
	cd machine
	rm *.png fit.log
	i_model=0
	# fitting
	while [ $i_model -lt $end_model ]; do
		i=0
		input_file="model.result."$i_model"."$i".npy"
		if [ ! -f $input_file ]; then
			let i_model=$i_model+1
			continue
		fi
		while [ $i -lt $n_files_train ]; do
			#echo "... fit and plot for " $i "th block"
			input_file="model.result."$i_model"."$i".npy"
			out_file="model.result."$i_model"."$i".png"
			python $path_py"machine/plot.py" -i $input_file -o $out_file >> fit.log
			let i=$i+1
		done
		let i_model=$i_model+1
	done
	# get average except abnormal values
	egrep --color 'Tc|Error:' fit.log | awk '{print $5}' > fit.value
	sed 's/found:/-1.0/g' fit.value > fit2.value
	python $path_py"machine/fit_avg.py" -i fit2.value -t1 $temp1 -t2 $temp2 -c $crit
	cd ..
	# returns "model.result."$i_model"."$i".png" files
	;;
* ) echo "wrong argument " $var_step
	;;
esac


