package com.veggie.forecast;

import java.util.List;
import java.util.Optional;
import org.springframework.data.repository.Repository;

interface ItemRepository extends Repository<Item, Long> {
    List<Item> findAllByOrderById();
    Optional<Item> findByName(String name);
}
