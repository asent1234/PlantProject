document.addEventListener('DOMContentLoaded', (event) => {
    const socket = io();
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatWindow = document.getElementById('chat-window');

    if (chatForm) {
        chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            if (chatInput.value) {
                socket.emit('send_message', chatInput.value);
                chatInput.value = '';
            }
        });
    }

    socket.on('receive_message', (data) => {
        const messageElement = document.createElement('p');
        messageElement.textContent = `${data.user}: ${data.message}`;
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    });
});