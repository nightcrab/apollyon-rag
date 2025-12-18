import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  	server: {
		proxy: {
		"/api": "http://localhost:8000"
		}
	},
	plugins: [sveltekit()]
});