# Enable terminal colors
export TERM=xterm-256color
force_color_prompt=yes

# Colored prompt - clearly indicates we're in a container
PS1='\[\033[01;35m\][container]\[\033[00m\] \[\033[01;32m\]\u\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '

# Enable color support for ls and grep
alias ls='ls --color=auto'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# Auto-activate Python virtual environment
if [ -d "$HOME/.venv" ]; then
    source "$HOME/.venv/bin/activate"
fi

# Useful ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

