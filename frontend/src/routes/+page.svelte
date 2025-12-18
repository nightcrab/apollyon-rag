<script>
	import { onMount, tick } from "svelte";

	let prompt = "";
	let loading = false;
	let textarea;
	let chatEl;

	let messages = [
		// { role: "user" | "assistant", content: "" }
	];

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


<style>
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

	.row.assistant {
		justify-content: flex-start;
	}

	.bubble {
		max-width: 720px;
		width: fit-content;
		padding: 0.85rem 1.1rem;
		border-radius: 14px;
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

	
	textarea {
		all: unset;
	}

	textarea::placeholder {
		color: #9ca3af;
	}

	button {
		all: unset;
	}

	.file-button {
		display: flex;
		justify-content: center;
		align-items: center;
	}

	.send-button {
		display: flex;
		justify-content: center;
		align-items: center;
		border-radius: 1rem;
		padding: 10px;
		border: none;
		background: rgb(20,20,20);
		color: white;
		font-weight: 500;
		cursor: pointer;
		transition: background 0.15s ease;
	}

	.send-button:hover:not(:disabled) {
		background: #000000;
	}

	.send-button:disabled {
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
		cursor: pointer;
	}

	.chat-textarea textarea {
		
		font-variant-ligatures: no-contextual;
		font-size: inherit;
		line-height: inherit;
		width: 100%;
		padding-bottom: 0;
		font-family: inherit;
		display: block;
	}
</style>

<div class="chat-container">
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
				<button class="file-button" type="submit" disabled={loading || !prompt.trim()}>
					<img src="/clip.svg" alt="File">
				</button>
				<textarea
					bind:this={textarea}
					bind:value={prompt}
					on:keydown={handleKeydown}
					rows="1"
					placeholder="Ask something…"
					disabled={loading}
				></textarea>
			</div>
			<div class="chat-options">
				<div class="spacer"></div>
			</div>
		</div>
	</form>
</div>
