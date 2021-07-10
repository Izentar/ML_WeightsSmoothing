#!/bin/bash 
#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

declare -a commands=(
    "python3 smoothing/experiments/exp_wide_resnet_disabled.py"
)

declare -a names=(
    "testModel"
)

: '
    "python smoothing/experiments/experiments_arithmeticMean.py"
    "python smoothing/experiments/experiments_borderline.py"
    "python smoothing/experiments/experiments_disabledMean.py"
    "python smoothing/experiments/experiments_movingMean.py"
    "python smoothing/experiments/experiments_weightedMean.py"
'

: '
    "python smoothing/alexnet_pretrain.py -l alexnet_pretrain_load -s alexnet_pretrain_save --mname alexnet_pretrain --test true --train true --pinTest false -d --debugOutput debug --modelOutput alexnet_pretrain_model --bashOutput true --formatedOutput alexnet_pretrain_formated"
    "python smoothing/alexnet.py -l alexnet_load -s alexnet_save --mname alexnet --test true --train true --pinTest false -d --debugOutput debug --modelOutput alexnet_model --bashOutput true --formatedOutput alexnet_formated"
    "python smoothing/googlenet.py -l googlenet_load -s googlenet_save --mname googlenet --test true --train true --pinTest false -d --debugOutput debug --modelOutput model --bashOutput true --formatedOutput googlenet_formated"
    "python smoothing/resnet18_pretrain.py -l resnet18_pretrain_load -s resnet18_pretrain_save --mname resnet18_pretrain --test true --train true --pinTest false -d --debugOutput debug --modelOutput resnet18_pretrain_model --bashOutput true --formatedOutput resnet18_pretrain_formated"
    "python smoothing/resnet18.py -l resnet18_load -s resnet18_save --mname resnet18 --test true --train true --pinTest false -d --debugOutput debug --modelOutput resnet18_model --bashOutput true --formatedOutput resnet18_formated"
    "python smoothing/vgg11_pretrain.py -l vgg11_pretrain_load -s vgg11_pretrain_save --mname vgg11_pretrain --test true --train true --pinTest false -d --debugOutput debug --modelOutput vgg11_pretrain_model --bashOutput true --formatedOutput vgg11_pretrain_formated"
    "python smoothing/vgg11.py -l vgg11_load -s vgg11_save --mname vgg11 --test true --train true --pinTest false -d --debugOutput debug --modelOutput vgg11_model --bashOutput true --formatedOutput vgg11_formated"
'

: '
    "python smoothing/alexnet_CIFAR100_disabledSmoothing.py -l alexnet_CIFAR100_disabledSmoothing_load -s alexnet_CIFAR100_disabledSmoothing_save --log alexnet_CIFAR100_disabledSmoothing --mname alexnet_CIFAR100_disabledSmoothing --test true --train true --pinTest false -d --bashOutput true"
    "python smoothing/alexnet_MINST_disabledSmoothing.py -l alexnet_MINST_disabledSmoothing_load -s alexnet_MINST_disabledSmoothing_save --log alexnet_MINST_disabledSmoothing --mname alexnet_MINST_disabledSmoothing --test true --train true --pinTest false -d --bashOutput true"
    "python smoothing/simpleModel_CIFAR100_disabledSmoothing.py -l simpleModel_CIFAR100_disabledSmoothing_load -s simpleModel_CIFAR100_disabledSmoothing_save --log simpleModel_CIFAR100_disabledSmoothing --mname simpleModel_CIFAR100_disabledSmoothing --test true --train true --pinTest false -d --bashOutput true"
    "python smoothing/simpleModel_MINST_disabledSmoothing.py -l simpleModel_MINST_disabledSmoothing_load -s simpleModel_MINST_disabledSmoothing_save --log simpleModel_MINST_disabledSmoothing --mname simpleModel_MINST_disabledSmoothing --test true --train true --pinTest false -d --bashOutput true"

    "python smoothing/alexnet_CIFAR100_generMean.py -l alexnet_CIFAR100_generMean_load -s alexnet_CIFAR100_generMean_save --log alexnet_CIFAR100_generMean --mname alexnet_CIFAR100_generMean --test true --train true --pinTest false -d --bashOutput true"
    "python smoothing/alexnet_CIFAR100_weightened.py -l alexnet_CIFAR100_weightened_load -s alexnet_CIFAR100_weightened_save --log alexnet_CIFAR100_weightened --mname alexnet_CIFAR100_weightened --test true --train true --pinTest false -d --bashOutput true"
    "python smoothing/alexnet_MINST_generMean.py -l alexnet_MINST_generMean_load -s alexnet_MINST_generMean_save --log alexnet_MINST_generMean --mname alexnet_MINST_generMean --test true --train true --pinTest false -d --bashOutput true"
    "python smoothing/alexnet_MINST_weightened.py -l alexnet_MINST_weightened_load -s alexnet_MINST_weightened_save --log alexnet_MINST_weightened --mname alexnet_MINST_weightened --test true --train true --pinTest false -d --bashOutput true"
    "python smoothing/simpleModel_CIFAR100_generMean.py -l simpleModel_CIFAR100_generMean_load -s simpleModel_CIFAR100_generMean_save --log simpleModel_CIFAR100_generMean --mname simpleModel_CIFAR100_generMean --test true --train true --pinTest false -d --bashOutput true"
    "python smoothing/simpleModel_CIFAR100_weightened.py -l simpleModel_CIFAR100_weightened_load -s simpleModel_CIFAR100_weightened_save --log simpleModel_CIFAR100_weightened --mname simpleModel_CIFAR100_weightened --test true --train true --pinTest false -d --bashOutput true"
    "python smoothing/simpleModel_MINST_generMean.py -l simpleModel_MINST_generMean_load -s simpleModel_MINST_generMean_save --log simpleModel_MINST_generMean --mname simpleModel_MINST_generMean --test true --train true --pinTest false -d --bashOutput true"
    "python smoothing/simpleModel_MINST_weightened.py -l simpleModel_MINST_weightened_load -s simpleModel_MINST_weightened_save --log simpleModel_MINST_weightened --mname simpleModel_MINST_weightened --test true --train true --pinTest false -d --bashOutput true"
'

fileBashStateName="savedBash"
bashLogFile="bash.log"
saveAndExit=0
exitOnly=0
enableRm=1

saveScriptState() {
    printf "BASH: Saving script state\n" &>> $bashLogFile
    printf "$counter\n" &>> $bashLogFile
    printf "$counter" > "$fileBashStateName"
    printf "BASH: Script state saved\n" &>> $bashLogFile
}

loadScriptState() {
    if [ -f "$fileBashStateName" ]; then
        printf "BASH: Loading script state\n" &>> $bashLogFile
        localCounter=($(<$fileBashStateName))
        printf "BASH: Script state loaded\n" &>> $bashLogFile
        return $localCounter
    else
        return "0"
    fi
}

exitScript() {
    now=$(date)
    printf "\nScript end: $now\n===========================================================\n" &>> $bashLogFile
    exit
}

kllChild() {
    printf "BASH: Caught SIGINT. Ending child process\n" &>> $bashLogFile
    exitOnly=1
    kill -SIGINT "$child" 2>/dev/null
    wait $child
    printf "BASH: Child process ended\n" &>> $bashLogFile
}

catchSIG() {
    printf "BASH: Caught SIGTSTP\n" &>> $bashLogFile
    saveAndExit=1
}

trap catchSIG SIGTSTP
trap kllChild SIGINT

: 'if  [ -n "$1" ] && [[ "$1" -eq "rmLogs" ]]; then
    if [[ "$enableRm" -eq 1 ]]; then
        rm *.log
        rm *.csv # może później usunąć, na razie do testów
        exit
    fi
    exit
fi
'

# main
#counter=0
$(loadScriptState)
counter=$?

if [[ $counter -eq 0 ]]; then
    printf "\n\nNew script call\n==================================================\n" &>> $bashLogFile
    now=$(date)
    printf "$now\n---------\n" &>> $bashLogFile
else
    printf "\n\nScript resuming\n==================================================\n" &>> $bashLogFile
    now=$(date)
    printf "Command number: $counter; date = $now\n---------\n" &>> $bashLogFile
fi

startLoop=$counter

for command in "${commands[@]:$startLoop} $@"
do
    printf "executing: $command\nFrom index: $counter starting at: $startLoop\n"
    $command &>> $bashLogFile &
    child=$!
    wait $child
    if [[ "$exitOnly" -eq 1 ]]; then
        exitScript
    elif [[ "$saveAndExit" -eq 1 ]]; then
        saveScriptState
        exitScript
    fi
    printf "executed: $command\nFrom index: $counter\n"
    ((++counter)) # trzeba by było się pokusić o to, aby python w pliku zapisywał, czy skończył trening i na tej podstawie aktualizować counter
                  # teraz jest mała szansa na to, że ponownie wykona ten sam skrypt.
done

let counter=0
echo "$counter" > "$fileBashStateName"
exitScript
python smoothing/playSound.py