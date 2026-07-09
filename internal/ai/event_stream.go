package ai

import (
	"context"
	"errors"
	"fmt"
)

var ErrStreamClosed = errors.New("event stream is closed")

type commandType uint8

const (
	commandPublish commandType = iota
	commandComplete
	commandFail
)

type command[T, R any] struct {
	typ    commandType
	event  T
	result R
	err    error
	reply  chan error
}

type EventStream[T, R any] struct {
	events chan T
	done   chan struct{}

	result R
	err    error
}

type Producer[T, R any] struct {
	commands chan<- command[T, R]
	done     <-chan struct{}
}

func NewEventStream[T, R any](
	ctx context.Context,
	buffer int,
) (*EventStream[T, R], *Producer[T, R]) {
	if ctx == nil {
		panic("nil context")
	}

	if buffer < 0 {
		panic("negative buffer size")
	}

	events := make(chan T, buffer)
	done := make(chan struct{})
	commands := make(chan command[T, R])

	stream := &EventStream[T, R]{
		events: events,
		done:   done,
	}

	producer := &Producer[T, R]{
		commands: commands,
		done:     done,
	}

	go stream.run(ctx, commands)

	return stream, producer
}

func (s *EventStream[T, R]) run(
	ctx context.Context,
	commands <-chan command[T, R],
) {
	defer close(s.events)
	defer close(s.done)

	for {
		select {
		case <-ctx.Done():
			s.err = ctx.Err()
			return

		case cmd := <-commands:
			switch cmd.typ {
			case commandPublish:
				s.publish(ctx, cmd)

			case commandComplete:
				s.result = cmd.result
				cmd.reply <- nil
				return

			case commandFail:
				s.err = cmd.err
				cmd.reply <- nil
				return

			default:
				cmd.reply <- fmt.Errorf(
					"unknown event stream command: %d",
					cmd.typ,
				)
			}
		}
	}
}

func (s *EventStream[T, R]) publish(
	ctx context.Context,
	cmd command[T, R],
) {
	select {
	case s.events <- cmd.event:
		cmd.reply <- nil

	case <-ctx.Done():
		s.err = ctx.Err()
		cmd.reply <- ctx.Err()
	}
}
func (s *EventStream[T, R]) Events() <-chan T {
	return s.events
}

// Result 等待事件流结束，并返回最终结果。
//
// Result 可以被多个 goroutine 调用。
// 与只能消费一次的结果 channel 不同，结果保存在 EventStream 内部。
func (s *EventStream[T, R]) Result(ctx context.Context) (R, error) {
	var zero R

	if ctx == nil {
		return zero, errors.New("nil context")
	}

	select {
	case <-ctx.Done():
		return zero, ctx.Err()

	case <-s.done:
		return s.result, s.err
	}
}

// Publish 向事件流发布一个事件。
//
// 如果缓冲区已满，并且没有消费者继续读取，该方法会阻塞，
// 从而形成背压。
func (p *Producer[T, R]) Publish(
	ctx context.Context,
	event T,
) error {
	return p.sendCommand(ctx, command[T, R]{
		typ:   commandPublish,
		event: event,
	})
}

// Complete 成功结束事件流，并设置最终结果。
func (p *Producer[T, R]) Complete(
	ctx context.Context,
	result R,
) error {
	return p.sendCommand(ctx, command[T, R]{
		typ:    commandComplete,
		result: result,
	})
}

// Fail 以错误状态结束事件流。
func (p *Producer[T, R]) Fail(
	ctx context.Context,
	err error,
) error {
	if err == nil {
		return errors.New("event stream failure error cannot be nil")
	}

	return p.sendCommand(ctx, command[T, R]{
		typ: commandFail,
		err: err,
	})
}

func (p *Producer[T, R]) sendCommand(
	ctx context.Context,
	cmd command[T, R],
) error {
	if ctx == nil {
		return errors.New("nil context")
	}

	// 必须使用缓冲 channel。
	//
	// 调用者发送命令后可能因 context 取消而提前返回。
	// 缓冲区可以防止内部 goroutine 在回复时永久阻塞。
	cmd.reply = make(chan error, 1)

	select {
	case <-ctx.Done():
		return ctx.Err()

	case <-p.done:
		return ErrStreamClosed

	case p.commands <- cmd:
	}

	select {
	case err := <-cmd.reply:
		return err

	case <-ctx.Done():
		return ctx.Err()

	case <-p.done:
		// Complete/Fail 的处理顺序是：
		// 先写 reply，再关闭 done。
		//
		// 因此 done 关闭时，reply 中通常已经有结果。
		select {
		case err := <-cmd.reply:
			return err
		default:
			return ErrStreamClosed
		}
	}
}
