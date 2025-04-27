<script lang="ts">
    export let label: string;
    export let value: boolean;
    export let onToggle: () => void;
  </script>
  
  <div class="flex items-center justify-between px-2 py-1 bg-black/20 rounded-md text-white text-xs whitespace-nowrap w-[9rem]">
    <span>{label}</span>
    <button
      on:click={onToggle}
      class="w-8 h-4 flex items-center rounded-full p-[2px] duration-300 ease-in-out"
      class:bg-green-500={value}
      class:bg-gray-600={!value}
    >
      <div
        class="bg-white w-3 h-3 rounded-full shadow-md transform duration-300 ease-in-out"
        class:translate-x-4={value}
      />
    </button>
  </div>
