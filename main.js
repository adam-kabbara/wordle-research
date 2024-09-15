const words = ['APPLE', 'BRAVE', 'CACTUS', 'DANCE', 'EAGLE', 'FROWN', 'GRAPE', 'HOUSE', 'IVORY', 'JELLY'];
const secretWord = words[Math.floor(Math.random() * words.length)];
const gameBoard = document.getElementById('game-board');
const keyboard = document.getElementById('keyboard');
const messageElement = document.getElementById('message');
let currentRow = 0;
let currentTile = 0;
const maxAttempts = 6;

// Initialize game board
for (let i = 0; i < maxAttempts; i++) {
    for (let j = 0; j < 5; j++) {
        const tile = document.createElement('div');
        tile.classList.add('tile');
        gameBoard.appendChild(tile);
    }
}

// Initialize keyboard
'QWERTYUIOPASDFGHJKLZXCVBNM'.split('').forEach(letter => {
    const key = document.createElement('div');
    key.classList.add('key');
    key.textContent = letter;
    key.addEventListener('click', () => handleInput(letter));
    keyboard.appendChild(key);
});

function handleInput(letter) {
    if (currentRow >= maxAttempts) return;
    if (currentTile < 5) {
        const tile = gameBoard.children[currentRow * 5 + currentTile];
        tile.textContent = letter;
        currentTile++;
    }
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        checkWord();
    } else if (e.key === 'Backspace') {
        deleteLetter();
    } else if (e.key.match(/^[a-zA-Z]$/)) {
        handleInput(e.key.toUpperCase());
    }
});

function deleteLetter() {
    if (currentTile > 0) {
        currentTile--;
        const tile = gameBoard.children[currentRow * 5 + currentTile];
        tile.textContent = '';
    }
}

function checkWord() {
    if (currentTile !== 5) return;

    const guess = Array.from(gameBoard.children)
        .slice(currentRow * 5, currentRow * 5 + 5)
        .map(tile => tile.textContent)
        .join('');

    const result = Array(5).fill('gray');
    const secretLetters = secretWord.split('');

    // Check for correct letters in correct positions
    for (let i = 0; i < 5; i++) {
        if (guess[i] === secretWord[i]) {
            result[i] = 'green';
            secretLetters[i] = null;
        }
    }

    // Check for correct letters in wrong positions
    for (let i = 0; i < 5; i++) {
        if (result[i] === 'green') continue;
        const index = secretLetters.indexOf(guess[i]);
        if (index !== -1) {
            result[i] = 'yellow';
            secretLetters[index] = null;
        }
    }

    // Apply colors to tiles
    for (let i = 0; i < 5; i++) {
        const tile = gameBoard.children[currentRow * 5 + i];
        tile.style.backgroundColor = result[i];
        tile.style.color = 'white';
    }

    if (guess === secretWord) {
        messageElement.textContent = 'Congratulations! You guessed the word!';
        currentRow = maxAttempts; // End the game
    } else if (currentRow === maxAttempts - 1) {
        messageElement.textContent = `Game over! The word was ${secretWord}.`;
    } else {
        currentRow++;
        currentTile = 0;
    }
}
