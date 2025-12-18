<script>
	import { onMount, tick } from "svelte";

	let prompt = "";
	let loading = false;
	let textarea;
	let chatEl;

	let uploading = false;

	let fileInput;

	let displayedError;

	let messages = [
		// { role: "user" | "assistant", content: "" }
	];

	async function handleFileChange(event) {

        const file = event.target.files[0];
        if (!file) return;

        try {
			console.log(file);
			uploading = true;
            await uploadFile(file);
			uploading = false;
			messages = [...messages, { role: "system", content: "Uploaded File: "+file.name }];
		} catch (error) {
            displayedError = error.message;
            console.error(error);
        }
	}

	async function uploadFile(blob) {

		const formData = new FormData();

		if (!!blob) {
			formData.append('file', blob);
		} else {
			throw new Error('No file to upload');
		}
		
		formData.append('session_id', 'default_session')

		const response = await fetch(
			`/api/upload`,
			{
				method: 'POST',
				headers: {
				},
				body: formData,
			}
		);

		const result = await response.json();

		if (!response.ok) {
			throw new Error(result.error || 'Failed to upload file');
		}

		return;
	}

	function openFile() {
        fileInput.click();
		console.log("clicked")
	}

	async function sendPrompt() {
		if (!prompt.trim() || loading) return;

		const userMessage = prompt;
		prompt = "";

		// 1. Add user message
		messages = [...messages, { role: "user", content: userMessage }];

		// 2. Add placeholder assistant message
		let assistantIndex = messages.length;
		messages = [...messages, { role: "assistant", content: "" }];

		messages[assistantIndex].loading = true;
		loading = true;

		await tick();
		scrollToBottom();

		const res = await fetch("/api/chat", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				prompt: userMessage,
				session_id: "default_session"
			})
		});

		const reader = res.body.getReader();
		const decoder = new TextDecoder();

		while (true) {
			const { value, done } = await reader.read();

			messages[assistantIndex].loading = false;
			loading = false;

			if (done) break;

			messages[assistantIndex].content += decoder.decode(value);
			messages = [...messages];
			await tick();
			scrollToBottom();
		}

	}

	function handleKeydown(e) {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			sendPrompt();
		}
	}


	function scrollToBottom() {
		if (chatEl) {
			chatEl.scrollTop = chatEl.scrollHeight;
		}
	}
</script>


<div class="chat-container">
	<input
		type="file"
		bind:this={fileInput}
		on:change={handleFileChange}
		accept="text/*"
		style="display: none;"
	/>

	<div class="messages" bind:this={chatEl}>
		{#each messages as msg}
			<div class="row {msg.role}">
				
				{#if msg.loading}
					<div class="bubble typing">Thinking…</div>
				{:else}
					<div class="bubble">
						{msg.content}
					</div>
				{/if}
			</div>
		{/each}

	</div>

	<form class="composer" on:submit|preventDefault={sendPrompt}>
		<div class="chat-input">
			<div class="chat-textarea">
				<button class="file-button" 
				on:click={openFile}>
					<img src="/clip.svg" alt="File">
				</button>
				<div class="text-area-container">
					<div
						class="text-area"
						bind:this={textarea}
						contenteditable="true"
						bind:textContent={prompt}
						on:keydown={handleKeydown}
						rows="1"
						
						disabled={loading}
						role="textbox"
						tabindex="0"
					>
					</div>
					{#if !prompt.trim()}
					<div class="placeholder">
						Ask anything...
					</div>
					{/if}
				</div>
				<button class="file-button" disabled={!prompt.trim() || loading}
				on:click={sendPrompt}>
					<img src="/send.svg" alt="File">
				</button>
			</div>
			<div class="chat-options">
				<div class="spacer"></div>
			</div>
		</div>
	</form>
	{#if displayedError}
	<div class="error-container">
		<div class="error">
			{displayedError}
		</div>
	</div>
	{/if}
	{#if uploading && !displayedError}
	<div class="error-container">
		<div class="error">
			Uploading file...
		</div>
	</div>
	{/if}
</div>


<style>

	.error-container {
		position: absolute;
		justify-content: center;
		align-items: top;
		width: 100vw;
		height: 100vh;
		top:0px;
		left:0px;	
		pointer-events: none;
	}

	.error {
		margin-top: 2em;
		padding:0.5em;
		display: flex;
		justify-content: center;
		align-items: center;
		border-radius: 1em;
	}

	@font-face {
		font-family: "Montserrat";
		src: url("/Montserrat.woff2") format("woff2");
	}


	.chat-container {
		font-family: "Montserrat", system-ui, -apple-system, BlinkMacSystemFont;
		max-width: 820px;
		margin: 2rem auto;
		height: 85vh;
		display: flex;
		flex-direction: column;
		overflow: hidden;
		font-weight: 450;
	}

	* {
		
		box-sizing: border-box;
	}
	
	.messages {
		flex: 1;
		overflow-y: auto;
		padding: 2rem 1.5rem;
		display: flex;
		flex-direction: column;
		gap: 1.25rem;
	}

	.row {
		display: flex;
	}

	.row.user {
		justify-content: flex-end;
	}
	
	.row.system {
		justify-content: center;
	}

	.row.assistant {
		justify-content: flex-start;
	}

	.bubble {
		max-width: 720px;
		width: fit-content;
		padding: 0.75em;
		border-radius: 1em;
		line-height: 1.6;
		font-size: 0.95rem;
		white-space: pre-wrap;
		word-break: break-word;
	}

	.row.user .bubble {
		background: rgb(240,240,240);
		color: black;
		border-bottom-right-radius: 6px;
	}

	.row.assistant .bubble {
	}

	/* Typing state */
	.typing {
		opacity: 0.6;
		font-style: italic;
	}

	/* ===== Composer ===== */

	.composer {
		display: flex;
		flex-direction: row;
		align-items: center;
		gap: 0.75rem;
		padding: 1rem;
		background: #ffffff;
	}

	
	.text-area {
		all: unset;
	}

	.placeholder {
		color: #9ca3af;
		position: absolute;
		pointer-events: none;
		top:0px;
		left:0px;
	}

	button {
		all: unset;
	}

	.file-button {
		display: flex;
		justify-content: center;
		align-items: center;
		cursor: pointer;
	}

	.file-button:disabled {
		opacity: 0.8;
		cursor: not-allowed;
	}
	
	.chat-input {
		flex: 1;
		resize: none;
		display: flex;
		flex-direction: column;
		padding: 0.9rem 1rem;
		border-radius: 1.5rem;
		border: 1px solid #d1d5db;
		font-family: inherit;
		font-size: 0.95rem;
		line-height: 1.5;
		max-height: 200px;
		overflow-y: auto;
		background: #ffffff;
	}

	.chat-options {
		display: flex;
		flex-direction: row;
	}

	.spacer {
		display: flex;
		flex: 1;
	}
	
	.chat-textarea {
		display: flex;
		flex: 1;
		width: 100%;
		gap: 1em;
	}
	
	.file-button img {
		display: inline;
		width: 1.5em;
		height: 1.5em;
	}

	.text-area-container {
		position: relative;
		width: 100%;
	}

	.chat-textarea .text-area {
		
		font-variant-ligatures: no-contextual;
		font-size: inherit;
		line-height: inherit;
		width: 100%;
		padding-bottom: 0;
		font-family: inherit;
		display: block;
	}
	.row {
		display: flex;
		opacity: 0;                  /* Start hidden */
		transform: translateY(20px); /* Start slightly below */
		transition: opacity 0.4s ease, transform 0.4s ease;
		animation: fadeSlideIn 0.4s ease forwards;
	}

	/* Keyframes for smooth entrance */
	@keyframes fadeSlideIn {
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}

	/* Optional: slightly different timing for user vs assistant for a more natural feel */
	.row.user {
		animation-delay: 0.05s;
	}

	.row.assistant {
		animation-delay: 0.1s;
	}

	.row.system {
		animation-delay: 0s;
	}

	/* For the streaming assistant message, make sure new content doesn't trigger re-animation */
	.row.assistant .bubble {
		transition: none; /* Prevent re-triggering animation when content updates */
	}
</style>
