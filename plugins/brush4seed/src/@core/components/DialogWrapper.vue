<template>
  <VDialog v-model="dialogModel" v-bind="$attrs" @update:model-value="handleDialogChange">
    <slot />
  </VDialog>
</template>

<script setup lang="ts">
import { computed, watch, onBeforeUnmount } from 'vue'
import { useScrollLockWithWatch } from '@/composables/useScrollLock'

// Props
interface Props {
  modelValue?: boolean
  // 滚动锁定配置
  scrollLock?: boolean
  preserveScrollPosition?: boolean
  preventTouchScroll?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  scrollLock: true,
  preserveScrollPosition: true,
  preventTouchScroll: true,
})

// Emits
const emit = defineEmits<{
  'update:modelValue': [value: boolean]
}>()

// 计算属性
const dialogModel = computed({
  get: () => props.modelValue || false,
  set: (value: boolean) => emit('update:modelValue', value),
})

// 使用滚动锁定
const { isLocked, lockScroll, restoreScroll } = useScrollLockWithWatch(dialogModel, {
  autoRestore: true,
  preserveScrollPosition: props.preserveScrollPosition,
  preventTouchScroll: props.preventTouchScroll,
})

// 处理弹窗状态变化
const handleDialogChange = (value: boolean) => {
  emit('update:modelValue', value)
}

// 监听弹窗状态变化
watch(
  dialogModel,
  newValue => {
    if (props.scrollLock) {
      if (newValue) {
        lockScroll()
      } else {
        restoreScroll()
      }
    }
  },
  { immediate: true },
)

// 组件卸载时确保恢复滚动
onBeforeUnmount(() => {
  if (isLocked.value) {
    restoreScroll()
  }
})
</script>
