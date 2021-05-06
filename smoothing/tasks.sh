#!/bin/bash 

declare -a commands=(
    "python smoothing/alexnet_pretrain.py -l test -s test --mname testModel --test true --train true --pinTest false -d --debugOutput debug --modelOutput model --bashOutput true --formatedOutput formated"
)

declare -a names=(
    "testModel"
)

#"python smoothing/convolutional.py -l test -s test --test on --train on --pinTest off -d"

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

if  [ -n "$1" ] && [[ "$1" -eq "rmLogs" ]]; then
    if [[ "$enableRm" -eq 1 ]]; then
        rm *.log
        rm *.csv # może później usunąć, na razie do testów
        exit
    fi
    exit
fi

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

for command in "${commands[@]:$startLoop}"
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