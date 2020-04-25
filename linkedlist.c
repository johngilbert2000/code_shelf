#include <stdio.h>
#include <stdlib.h>
#include <string.h>


struct linked {
    int val;
    struct linked *next;
    struct linked *prev;
};

typedef struct linked node;

node *create_node(int val){
    node *res = malloc(sizeof(node));
    res->val = val;
    res->next = NULL;
    res->prev = NULL;
    return res;
}

void insert_node(node *previous_node, node *new_node){
    // Inserts a new node after previous node
    new_node->next = previous_node->next;
    if (new_node->next != NULL) {
        new_node->next->prev = new_node;
    }
    new_node->prev = previous_node;
    previous_node->next = new_node;
}

void append_value(node **tail, int val) {
    // appends node with given value after tail
    // sets the pointer tail = new_node, if tail was the end of the list
    // Example: append_value(&tail, 2)  (adds node, and now tail->val == 2)
    node *tmp;
    tmp = create_node(val);
    insert_node(*tail, tmp);
    if (tmp->next == NULL) {
        *tail = tmp;
    }
}

void insert_value(node *previous_node, int val){
    // Inserts node with given value after previous_node
    insert_node(previous_node, create_node(val));
}

node *insert_head(node **head, node *new_node){
    // Inserts a new node at the head of the linked list
    new_node->next = *head;
    if (*head != NULL) {
        (*head)->prev = new_node;
    }
    *head = new_node;
    new_node->prev = NULL;
    return new_node;
}

void remove_node(node *old_node){
    // Removes node from linked list
    if (old_node->prev != NULL) old_node->prev->next = old_node->next;
    if (old_node->next != NULL) old_node->next->prev = old_node->prev;
    old_node->prev = NULL;
    old_node->next = NULL;
}

node *get_head(node *any_node){
    node *head;
    head = any_node;
    while (any_node->prev != NULL){
        head = any_node->prev;
        any_node = any_node->prev;
    }
    return head;
}

node *get_tail(node *any_node){
    node *tail;
    tail = any_node;
    while (any_node->next != NULL) {
        tail = any_node->next;
        any_node = any_node->next;
    }
    return tail;
}

void remove_head(node **head){
    // Removes head from linked list, and sets *head equal to next node in list
    // Example: remove_head(&head);
    *head = get_head(*head); // ensures that *head points to actual head node
    node *old_head;
    old_head = *head;
    if (old_head->next != NULL) *head = old_head->next;
    remove_node(old_head);
}

void remove_tail(node **tail){
    // Sets the tail node equal to the second last node, then removes the old tail
    // Example: remove_tail(&tail);
    *tail = get_tail(*tail); // ensures that *tail points to actual tail node
    node *old_tail;
    old_tail = *tail;
    if (old_tail->prev != NULL) *tail = old_tail->prev;
    remove_node(old_tail);
}

node *lookup(node *head, int val) {
    // Obtains the first node with the given value
    node *tmp = head;
    while (tmp != NULL) {
        if (tmp->val == val) return tmp;
        tmp = tmp->next;
    }
    return NULL;
}

void print_list(node *head){
    node *tmp = head;
    printf("(");
    while (tmp->next != NULL) {
        printf("%d, ", tmp->val);
        tmp = tmp->next;
    }
    if (tmp != NULL) printf("%d", tmp->val);
    printf(")\n");
}

void print_reverse(node *tail){
    // print the linked list in reverse
    node *tmp = tail;
    printf("(");
    while (tmp->prev != NULL) {
        printf("%d, ", tmp->val);
        tmp = tmp->prev;
    }
    if (tmp != NULL) printf("%d", tmp->val);
    printf(")\n");
}

void print_node(node *any_node){
    printf("%d", any_node->val);
    printf("\n");
}

int main() {
    node *head;
    node *tmp;
    node *tail;
    node *tmp2;

    tmp = create_node(1);
    head = tmp;
    tail = tmp;

    append_value(&tail, 2);
    append_value(&tail, 3);
    append_value(&tail, 4);
    append_value(&tail, 5);
    append_value(&tail, 7);
    append_value(&tail, 8);
    append_value(&tail, 9);
    append_value(&tail, 10);


    // tmp = create_node(4);
    // head = tmp;
    // tail = head;

    // tmp = create_node(5);
    // insert_node(tail, tmp);
    // tail = tmp;

    // tmp = create_node(7);
    // insert_node(tail, tmp);
    // tail = tmp;

    // append_value(&tail, 9);
    // // insert_value(tmp, 8);
    // append_value(&tmp, 8);

    print_list(head);

    tmp = lookup(head, 2);
    remove_node(tmp);

    tmp = lookup(head, 5);
    printf("lookup: %d\n", tmp->val);
    printf("previous: %d \n", tmp->prev->val);
    printf("next: %d \n", tmp->next->val);

    append_value(&tmp, 6);

    // tmp2 = create_node(6);
    // insert_node(tmp, tmp2);
   
    print_list(head);
    print_reverse(tail);

    remove_head(&head);
    print_list(head);

    remove_head(&head);
    print_list(head);


    tmp = lookup(head, 9);
    tmp2 = get_head(tmp);
    print_node(tmp2);
    tmp2 = get_tail(tmp);
    print_node(tmp2);

    remove_tail(&tail);
    print_list(head);
    print_node(tail);

    return 0;
}